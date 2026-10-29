"""Energy-based first-pass event gate.

Pure library code: no I/O, no plotting, no CLI, no printing, no logging.
Streaming by design — :class:`EnergyGate` consumes one short block of
samples at a time via :meth:`EnergyGate.process_block`, so callers can feed
it an 8-hour recording without ever holding more than a few seconds of
audio in memory (see ``tools/detect_events.py``).

## Why energy in a frequency band, not overall energy

Sprint 1 measured 15.6 dB SNR for grinding vs. silence in the 8-16 kHz band,
while snoring/HVAC/room rumble sit below ~1 kHz and account for >90% of
total signal energy. Gating on band-limited energy throws out that
dominant low-frequency noise before any modeling happens.

## Why the band is a parameter, not a constant

Sprint 1's 8-16 kHz figure came from deliberate tooth-*rubbing* recorded
~30 cm from the mic. Clips of actual sleep bruxism (confirmed by ear) look
very different: 0.1-2.4% of energy above 8 kHz instead of 32-47%, spectral
centroid ~3 kHz instead of ~9-10 kHz. Distance and bedding absorption
plausibly eat the top octaves, which would mean the original
2000-16000 Hz band is largely listening where the real signal isn't.
``low_hz``/``high_hz`` are therefore caller-supplied so the band can be
swept against real data (``tools/sweep_band.py``) rather than assumed, and
every emitted :class:`EventCandidate` records the band it came from.

## Why a causal (not zero-phase) filter

``src.features.spectral.high_pass_filter`` uses ``sosfiltfilt`` (zero-phase,
whole-signal) because it operates on short pilot clips already fully in
memory. That's incompatible with streaming an 8-hour file block-by-block.
This module uses ``sosfilt`` with filter state (``zi``) carried across
blocks instead — mathematically equivalent to filtering the whole signal
at once, just causal, so it introduces a small constant group delay
(a few ms at these frequencies/order) instead of zero phase shift. That
delay does not matter for event *detection* (timing accuracy to within a
frame hop is more than enough) and is documented here rather than hidden.

## Why the adaptive threshold recomputes every ~1s, not every frame

`threshold = median + k * MAD` over a trailing 5-minute window is exactly
what the spec calls for, but recomputing it from a ~12,000-sample rolling
window on every 25ms frame (up to ~1.15M times for an 8-hour night) is
the difference between a run finishing in seconds vs. minutes. The
background noise floor drifts on the order of minutes, not milliseconds,
so recomputing once per second of audio tracks it just as well at a
fraction of the cost. This cadence is counted in *frames*, not tied to
however big a block the caller passes to `process_block` — otherwise the
caller's I/O chunk size would silently change detection results.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal as sp_signal

_EPS = 1e-12

FRAME_MS = 50.0
HOP_MS = 25.0
ADAPTIVE_WINDOW_S = 300.0  # 5 minutes trailing
THRESHOLD_RECOMPUTE_INTERVAL_S = 1.0
DEFAULT_K = 6.0
HYSTERESIS_EXIT_RATIO = 0.6
MERGE_GAP_S = 0.5
DEFAULT_MIN_EVENT_DURATION_S = 0.4
DEFAULT_LOW_HZ = 2000.0
DEFAULT_HIGH_HZ = 16000.0
HIGH_HZ_NYQUIST_MARGIN = 0.95
NARROW_BAND_WARNING_HZ = 6000.0
MAD_FLOOR = 1e-6  # linear amplitude; prevents threshold collapsing to the median on near-silent audio
FILTER_ORDER = 4


@dataclass(frozen=True)
class GateBand:
    """The frequency band actually used for one sample rate.

    ``low_hz``/``high_hz`` are the *effective* edges after Nyquist capping,
    not necessarily what the caller requested — ``requested_high_hz`` keeps
    the original ask so a tool can tell the user their 16 kHz upper edge was
    silently reduced on a low-sample-rate file.
    """

    low_hz: float
    high_hz: float
    narrow_band_warning: bool
    requested_high_hz: float

    @property
    def mid_hz(self) -> float:
        """Split point between the band's lower and upper half.

        This is what ``EventCandidate.band_ratio_high`` is measured against,
        so it moves with the band: narrowing the band to 1000-6000 Hz makes
        the ratio describe 3500-6000 Hz, not the ~9 kHz split the default
        2000-16000 Hz band implies.
        """
        return (self.low_hz + self.high_hz) / 2.0

    @property
    def high_hz_was_capped(self) -> bool:
        return self.high_hz < self.requested_high_hz


def resolve_band(
    sample_rate: int, low_hz: float = DEFAULT_LOW_HZ, high_hz: float = DEFAULT_HIGH_HZ
) -> GateBand:
    """Pick the gate's frequency band for a given sample rate.

    The requested ``high_hz`` is additionally capped at 95% of Nyquist — the
    same margin ``src.features.spectral`` implicitly assumes by never
    analyzing all the way to Nyquist.

    Raises:
        ValueError: if ``high_hz <= low_hz`` as requested, or if Nyquist
            capping leaves no usable band at this sample rate (the default
            2 kHz low edge needs roughly sample_rate > ~4.2 kHz).
    """
    if high_hz <= low_hz:
        raise ValueError(f"ย่านความถี่ไม่ถูกต้อง: high_hz ({high_hz:.0f}) ต้องมากกว่า low_hz ({low_hz:.0f})")
    nyquist = sample_rate / 2.0
    effective_high_hz = min(high_hz, nyquist * HIGH_HZ_NYQUIST_MARGIN)
    if effective_high_hz <= low_hz:
        raise ValueError(
            f"sample rate {sample_rate} Hz ต่ำเกินไปสำหรับย่าน {low_hz:.0f} Hz ขึ้นไป "
            f"(Nyquist * {HIGH_HZ_NYQUIST_MARGIN:.2f} = {effective_high_hz:.0f} Hz)"
        )
    return GateBand(
        low_hz=low_hz,
        high_hz=effective_high_hz,
        narrow_band_warning=(effective_high_hz - low_hz) < NARROW_BAND_WARNING_HZ,
        requested_high_hz=high_hz,
    )


@dataclass
class EventCandidate:
    event_id: str
    session_id: str
    start_s: float
    end_s: float
    duration_s: float
    peak_rms_db: float
    mean_rms_db: float
    snr_above_floor_db: float
    band_ratio_high: float  # energy above band_mid_hz / energy in the whole gate band
    segment_file: str
    # The detection settings this event came out of, carried per-row so two
    # runs over the same night (different band or k) stay comparable after
    # the fact — a bare candidates.csv otherwise says nothing about how it
    # was produced. These are the *effective* values after Nyquist capping.
    band_low_hz: float
    band_high_hz: float
    band_mid_hz: float
    gate_k: float


@dataclass
class _RawEvent:
    start_s: float
    end_s: float
    peak_rms_lin: float
    sum_full_power: float
    sum_upper_power: float
    n_frames: int
    floor_db: float
    segment_file: str


class _StreamingFramer:
    """Turns a stream of filtered sample blocks into fixed-size overlapping frames.

    Carries whatever trailing samples didn't yet complete a frame between
    calls, so framing is identical to running on the whole concatenated
    signal at once regardless of how it's chunked into blocks.
    """

    def __init__(self, frame_len: int, hop_len: int):
        self.frame_len = frame_len
        self.hop_len = hop_len
        self._leftover = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> np.ndarray:
        combined = np.concatenate([self._leftover, block])
        if combined.size < self.frame_len:
            self._leftover = combined
            return np.empty((0, self.frame_len), dtype=combined.dtype)
        windows = sliding_window_view(combined, self.frame_len)[:: self.hop_len]
        n_frames = windows.shape[0]
        self._leftover = combined[n_frames * self.hop_len :]
        return windows


class EnergyGate:
    """Streaming energy-in-band event gate.

    Usage::

        gate = EnergyGate(sample_rate=44100, session_id="ext_20260820")
        for block, segment_file in blocks:
            gate.process_block(block, segment_file)
        events = gate.finalize()

    ``block`` is a 1-D float array (mono, roughly [-1, 1]); blocks must be
    fed in chronological order with no gaps (see the module docstring and
    ``tools/detect_events.py`` for how segment boundaries / interruptions
    are handled — this class only ever sees a single continuous stream).

    After processing, ``envelope_trace_db`` and ``threshold_history_db``
    hold one ``(time_s, value_db)`` point per block — a cheap, plot-ready
    summary of the whole run for an overview chart, without needing to
    retain any of the underlying audio.
    """

    def __init__(
        self,
        sample_rate: int,
        session_id: str,
        k: float = DEFAULT_K,
        min_duration_s: float = DEFAULT_MIN_EVENT_DURATION_S,
        band_low_hz: float = DEFAULT_LOW_HZ,
        band_high_hz: float = DEFAULT_HIGH_HZ,
    ):
        self.sample_rate = sample_rate
        self.session_id = session_id
        self.k = k
        self.min_duration_s = min_duration_s
        self.band = resolve_band(sample_rate, band_low_hz, band_high_hz)

        self._sos_full = sp_signal.butter(
            FILTER_ORDER, [self.band.low_hz, self.band.high_hz], btype="bandpass", fs=sample_rate, output="sos"
        )
        self._sos_upper = sp_signal.butter(
            FILTER_ORDER, [self.band.mid_hz, self.band.high_hz], btype="bandpass", fs=sample_rate, output="sos"
        )
        self._zi_full: np.ndarray | None = None
        self._zi_upper: np.ndarray | None = None

        frame_len = int(round(FRAME_MS / 1000.0 * sample_rate))
        hop_len = int(round(HOP_MS / 1000.0 * sample_rate))
        self._hop_s = hop_len / sample_rate
        self._framer_full = _StreamingFramer(frame_len, hop_len)
        self._framer_upper = _StreamingFramer(frame_len, hop_len)

        window_frames = max(1, int(round(ADAPTIVE_WINDOW_S / self._hop_s)))
        self._rolling_rms: deque[float] = deque(maxlen=window_frames)

        # Recompute cadence is frame-count-based (not tied to how big a
        # block the caller happens to pass to process_block) so detection
        # results don't depend on the caller's I/O chunk size.
        self._threshold_recompute_interval_frames = max(1, int(round(THRESHOLD_RECOMPUTE_INTERVAL_S / self._hop_s)))
        self._cached_threshold_lin = float("inf")
        self._cached_median_lin = 0.0

        self._frame_index = 0
        self._in_event = False
        self._event_start_s = 0.0
        self._event_peak_lin = 0.0
        self._event_sum_full_power = 0.0
        self._event_sum_upper_power = 0.0
        self._event_n_frames = 0
        self._event_floor_db = 0.0
        self._event_segment_file = ""

        self._raw_events: list[_RawEvent] = []

        self.envelope_trace_db: list[tuple[float, float]] = []
        self.threshold_history_db: list[tuple[float, float]] = []

    def process_block(self, block: np.ndarray, segment_file: str) -> None:
        """Feed the next chunk of continuous audio (chronological order, no gaps)."""
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            return

        if self._zi_full is None:
            self._zi_full = sp_signal.sosfilt_zi(self._sos_full) * block[0]
            self._zi_upper = sp_signal.sosfilt_zi(self._sos_upper) * block[0]

        filtered_full, self._zi_full = sp_signal.sosfilt(self._sos_full, block, zi=self._zi_full)
        filtered_upper, self._zi_upper = sp_signal.sosfilt(self._sos_upper, block, zi=self._zi_upper)

        frames_full = self._framer_full.push(filtered_full.astype(np.float32))
        frames_upper = self._framer_upper.push(filtered_upper.astype(np.float32))
        n_frames = frames_full.shape[0]
        if n_frames == 0:
            return

        frame_rms_full = np.sqrt(np.mean(np.square(frames_full, dtype=np.float64), axis=1))
        frame_rms_upper = np.sqrt(np.mean(np.square(frames_upper, dtype=np.float64), axis=1))

        block_time_s = self._frame_index * self._hop_s
        block_rms_lin = float(np.sqrt(np.mean(np.square(frame_rms_full, dtype=np.float64))))
        self.envelope_trace_db.append((block_time_s, 20.0 * np.log10(max(block_rms_lin, _EPS))))

        for i in range(n_frames):
            t = self._frame_index * self._hop_s
            rms = float(frame_rms_full[i])
            rms_upper = float(frame_rms_upper[i])

            if self._frame_index % self._threshold_recompute_interval_frames == 0:
                self._cached_threshold_lin, self._cached_median_lin = self._compute_threshold()
                self.threshold_history_db.append(
                    (t, 20.0 * np.log10(max(self._cached_threshold_lin, _EPS)))
                )
            threshold_lin = self._cached_threshold_lin
            # A hysteresis band between the noise floor (median) and the
            # entry threshold — NOT a flat fraction of the entry threshold
            # itself. When k*MAD is a small fraction of the median (as it
            # legitimately is for fairly stationary background noise),
            # `threshold * ratio` can fall below the median entirely,
            # meaning ordinary background frames would never register as
            # "below exit" and an event, once started, would never end.
            exit_threshold_lin = self._cached_median_lin + HYSTERESIS_EXIT_RATIO * (
                threshold_lin - self._cached_median_lin
            )

            if self._in_event and rms < exit_threshold_lin:
                self._flush_event(end_s=t)

            if not self._in_event and rms > threshold_lin:
                self._start_event(start_s=t, segment_file=segment_file, floor_lin=self._cached_median_lin)

            if self._in_event:
                self._event_peak_lin = max(self._event_peak_lin, rms)
                self._event_sum_full_power += rms * rms
                self._event_sum_upper_power += rms_upper * rms_upper
                self._event_n_frames += 1

            self._rolling_rms.append(rms)
            self._frame_index += 1

    def finalize(self) -> list[EventCandidate]:
        """Flush any still-open event and return the final, merged, filtered event list."""
        if self._in_event:
            self._flush_event(end_s=self._frame_index * self._hop_s)

        merged = self._merge_close_events(self._raw_events)
        surviving = [e for e in merged if (e.end_s - e.start_s) >= self.min_duration_s]
        return [self._to_event_candidate(e, i) for i, e in enumerate(surviving)]

    def _compute_threshold(self) -> tuple[float, float]:
        if not self._rolling_rms:
            return float("inf"), 0.0  # no history yet — cold start, never trigger
        arr = np.fromiter(self._rolling_rms, dtype=np.float64, count=len(self._rolling_rms))
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        mad = max(mad, MAD_FLOOR)
        return median + self.k * mad, median

    def _start_event(self, start_s: float, segment_file: str, floor_lin: float) -> None:
        self._in_event = True
        self._event_start_s = start_s
        self._event_segment_file = segment_file
        self._event_peak_lin = 0.0
        self._event_sum_full_power = 0.0
        self._event_sum_upper_power = 0.0
        self._event_n_frames = 0
        self._event_floor_db = 20.0 * np.log10(max(floor_lin, _EPS))

    def _flush_event(self, end_s: float) -> None:
        self._raw_events.append(
            _RawEvent(
                start_s=self._event_start_s,
                end_s=end_s,
                peak_rms_lin=self._event_peak_lin,
                sum_full_power=self._event_sum_full_power,
                sum_upper_power=self._event_sum_upper_power,
                n_frames=self._event_n_frames,
                floor_db=self._event_floor_db,
                segment_file=self._event_segment_file,
            )
        )
        self._in_event = False

    @staticmethod
    def _merge_close_events(events: list[_RawEvent]) -> list[_RawEvent]:
        if not events:
            return []
        merged = [events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if ev.start_s - last.end_s < MERGE_GAP_S:
                merged[-1] = _RawEvent(
                    start_s=last.start_s,
                    end_s=ev.end_s,
                    peak_rms_lin=max(last.peak_rms_lin, ev.peak_rms_lin),
                    sum_full_power=last.sum_full_power + ev.sum_full_power,
                    sum_upper_power=last.sum_upper_power + ev.sum_upper_power,
                    n_frames=last.n_frames + ev.n_frames,
                    floor_db=last.floor_db,
                    segment_file=last.segment_file,
                )
            else:
                merged.append(ev)
        return merged

    def _to_event_candidate(self, raw: _RawEvent, index: int) -> EventCandidate:
        n = max(raw.n_frames, 1)
        mean_rms_lin = float(np.sqrt(raw.sum_full_power / n))
        peak_rms_db = 20.0 * np.log10(max(raw.peak_rms_lin, _EPS))
        mean_rms_db = 20.0 * np.log10(max(mean_rms_lin, _EPS))
        band_ratio_high = float(raw.sum_upper_power / raw.sum_full_power) if raw.sum_full_power > 0 else 0.0
        return EventCandidate(
            event_id=f"{self.session_id}_{index:04d}",
            session_id=self.session_id,
            start_s=raw.start_s,
            end_s=raw.end_s,
            duration_s=raw.end_s - raw.start_s,
            peak_rms_db=peak_rms_db,
            mean_rms_db=mean_rms_db,
            snr_above_floor_db=mean_rms_db - raw.floor_db,
            band_ratio_high=band_ratio_high,
            segment_file=raw.segment_file,
            band_low_hz=self.band.low_hz,
            band_high_hz=self.band.high_hz,
            band_mid_hz=self.band.mid_hz,
            gate_k=self.k,
        )
