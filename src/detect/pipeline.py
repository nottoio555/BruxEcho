"""Whole-night detection: energy gate → YAMNet → classifier → events.

Pure library code: no plotting, no CLI, no printing, no logging.

## Why the energy gate stays in front of the model

Running YAMNet over eight hours of audio is pointless work: most of a night
is background. The gate (Sprint 3) throws out ~95% of it on signal energy
alone in about a minute, and the classifier only ever sees what survives.

The gate is deliberately run at a *permissive* setting here. Sprint 3.5 tuned
it to reduce the number of clips a human had to listen to, which meant
trading recall away; now that a classifier does the discrimination, the gate's
only job is to not lose anything, so a low ``k`` is the right default.
Anything the gate drops is invisible to everything downstream.

## Why frames come straight from YAMNet

YAMNet already emits one embedding per 0.96s window with a 0.48s hop —
exactly the windowing this pipeline needs — so feeding it a candidate region
yields the frames directly, and frame ``i`` starts at ``region_start +
i * 0.48``. Re-windowing by hand would only risk disagreeing with the model's
own framing.

## Why median filter then hysteresis

Per-frame probabilities are jumpy: a single frame crossing the threshold is
usually noise, not a 0.48s grind. A short median filter removes those spikes
without shifting event boundaries the way a mean would. Hysteresis then uses
a lower threshold to *leave* an event than to *enter* one, so probabilities
hovering near the threshold produce one event instead of a burst of
fragments — the same reasoning (and the same failure mode) as the energy
gate's own hysteresis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

YAMNET_HOP_S = 0.48
YAMNET_WINDOW_S = 0.96

DEFAULT_THRESHOLD = 0.5
HYSTERESIS_EXIT_RATIO = 0.6
# 1 = no smoothing, and that is deliberate. Two reasons:
#
# 1. It matches how the threshold was chosen. `tools/evaluate_model` scores an
#    event by the peak of its *raw* frames, so shipping a median filter would
#    make the deployed detector strictly stricter than the one the 0.96 came
#    from — the evaluation would describe a detector that does not exist.
# 2. Measured cost on real labels: the night's confirmed grind at 16160.2s
#    peaks at 0.806 on a single frame with 0.098 and 0.166 either side. A
#    3-frame median turns that into 0.166 and deletes a true positive. A
#    3-frame median needs two consecutive frames over the threshold, i.e.
#    ~1.4s of sustained grinding, and real bursts are often shorter.
#
# YAMNet frames are already smoothed anyway: a 0.96s window every 0.48s means
# adjacent frames share half their audio. Raise this only if a recording is
# noisy enough that isolated spikes dominate.
DEFAULT_MEDIAN_FRAMES = 1
MEDIAN_FILTER_FRAMES = 3  # kept as the documented "smoothing on" setting
DEFAULT_MIN_EVENT_S = 0.5
DEFAULT_MERGE_GAP_S = 0.5
REGION_CONTEXT_S = 0.5  # widen each gate candidate so an onset isn't clipped mid-event

# Permissive gate defaults — see module docstring.
DEFAULT_GATE_K = 6.0
DEFAULT_GATE_LOW_HZ = 800.0
DEFAULT_GATE_HIGH_HZ = 4000.0
DEFAULT_READ_BLOCK_S = 2.0


@dataclass(frozen=True)
class Event:
    """One detected event in whole-night time coordinates."""

    start_s: float
    end_s: float
    label: str
    confidence: float  # mean probability of `label` across the event's frames

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Region:
    """A stretch of the night the gate thought was worth looking at."""

    start_s: float
    end_s: float


def median_filter_1d(values: np.ndarray, size: int = MEDIAN_FILTER_FRAMES) -> np.ndarray:
    """Odd-length median filter with edge padding (keeps length and alignment)."""
    if size <= 1 or values.size == 0:
        return values
    if size % 2 == 0:
        size += 1
    half = size // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=1).astype(values.dtype)


def merge_regions(regions: list[Region], gap_s: float = 1.0) -> list[Region]:
    """Merge overlapping/adjacent regions so audio isn't embedded twice."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.start_s)
    merged = [ordered[0]]
    for region in ordered[1:]:
        last = merged[-1]
        if region.start_s - last.end_s <= gap_s:
            merged[-1] = Region(last.start_s, max(last.end_s, region.end_s))
        else:
            merged.append(region)
    return merged


def frames_to_events(
    probabilities: np.ndarray,
    class_names: list[str],
    region_start_s: float,
    threshold: float,
    target_indices: list[int],
    min_event_s: float = DEFAULT_MIN_EVENT_S,
    hop_s: float = YAMNET_HOP_S,
    window_s: float = YAMNET_WINDOW_S,
    median_frames: int = DEFAULT_MEDIAN_FRAMES,
) -> list[Event]:
    """Turn a region's per-frame probabilities into events.

    Each target class is tracked independently, so an ambiguous stretch can
    surface as both a bruxism and a snore candidate rather than one silently
    suppressing the other.

    ``median_frames`` is the knob that decides how short an event can be. A
    3-frame median needs *two consecutive* frames over the threshold, i.e.
    roughly 1.4s of sustained grinding, which measured on the one night with
    human labels deletes real events: their bruxism peaks (0.94, 0.87) sit on
    single frames. Passing 1 disables the filter, which is defensible because
    YAMNet frames are already smoothed — a 0.96s window every 0.48s means
    adjacent frames share half their audio. The cost is that a lone spike can
    become an event, so ``min_event_s`` and the threshold carry that load
    instead.
    """
    events: list[Event] = []
    if probabilities.size == 0:
        return events

    exit_threshold = threshold * HYSTERESIS_EXIT_RATIO
    for class_index in target_indices:
        smoothed = median_filter_1d(probabilities[:, class_index], size=median_frames)
        in_event = False
        start_frame = 0
        for i, value in enumerate(smoothed):
            if not in_event and value >= threshold:
                in_event, start_frame = True, i
            elif in_event and value < exit_threshold:
                events.append(
                    _make_event(
                        smoothed, start_frame, i - 1, class_names[class_index], region_start_s, hop_s, window_s
                    )
                )
                in_event = False
        if in_event:
            events.append(
                _make_event(
                    smoothed,
                    start_frame,
                    len(smoothed) - 1,
                    class_names[class_index],
                    region_start_s,
                    hop_s,
                    window_s,
                )
            )

    return [e for e in events if e.duration_s >= min_event_s]


def _make_event(
    smoothed: np.ndarray,
    start_frame: int,
    end_frame: int,
    label: str,
    region_start_s: float,
    hop_s: float,
    window_s: float,
) -> Event:
    start_s = region_start_s + start_frame * hop_s
    # The last frame still covers a full window, so the event ends where that
    # window ends rather than where it starts.
    end_s = region_start_s + end_frame * hop_s + window_s
    confidence = float(np.mean(smoothed[start_frame : end_frame + 1]))
    return Event(start_s=start_s, end_s=end_s, label=label, confidence=confidence)


def merge_events(events: list[Event], gap_s: float = DEFAULT_MERGE_GAP_S) -> list[Event]:
    """Merge same-label events separated by less than ``gap_s``."""
    if not events:
        return []
    merged: list[Event] = []
    for event in sorted(events, key=lambda e: (e.label, e.start_s)):
        if merged and merged[-1].label == event.label and event.start_s - merged[-1].end_s <= gap_s:
            last = merged[-1]
            total = last.duration_s + event.duration_s
            merged[-1] = Event(
                start_s=last.start_s,
                end_s=max(last.end_s, event.end_s),
                label=last.label,
                # Duration-weighted so a long confident stretch isn't dragged
                # down by a short marginal one merged onto its end.
                confidence=(last.confidence * last.duration_s + event.confidence * event.duration_s)
                / max(total, 1e-9),
            )
        else:
            merged.append(event)
    return sorted(merged, key=lambda e: e.start_s)


def gate_regions(
    session_dir: Path,
    session,
    sample_rate: int,
    k: float = DEFAULT_GATE_K,
    low_hz: float = DEFAULT_GATE_LOW_HZ,
    high_hz: float = DEFAULT_GATE_HIGH_HZ,
    context_s: float = REGION_CONTEXT_S,
    read_block_s: float = DEFAULT_READ_BLOCK_S,
    on_progress=None,
) -> tuple[list[Region], float]:
    """Run the energy gate and return ``(regions, total_duration_s)``."""
    import soundfile as sf  # noqa: PLC0415 — lazy

    from src.detect.energy_gate import EnergyGate  # noqa: PLC0415

    gate = EnergyGate(
        sample_rate=sample_rate,
        session_id=getattr(session, "session_id", "session"),
        k=k,
        band_low_hz=low_hz,
        band_high_hz=high_hz,
    )
    block_frames = int(read_block_s * sample_rate)
    total_s = 0.0
    for seg in session.segments:
        path = session_dir / seg.file
        if not path.exists():
            continue
        for block in sf.blocks(str(path), blocksize=block_frames, dtype="float32", always_2d=False):
            audio = np.asarray(block, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            gate.process_block(audio, segment_file=seg.file)
            total_s += audio.size / sample_rate
        if on_progress is not None:
            on_progress(seg.file, total_s)

    regions = [
        Region(max(0.0, c.start_s - context_s), min(total_s, c.end_s + context_s))
        for c in gate.finalize()
    ]
    return merge_regions(regions), total_s


def read_window(
    session_dir: Path, session, start_s: float, end_s: float, sample_rate: int
) -> np.ndarray:
    """Mono audio for ``[start_s, end_s)`` of the night, spanning segments."""
    import soundfile as sf  # noqa: PLC0415

    chunks: list[np.ndarray] = []
    cursor = 0.0
    for seg in session.segments:
        path = session_dir / seg.file
        if not path.exists():
            continue
        info = sf.info(str(path))
        seg_duration = info.frames / info.samplerate
        seg_start, seg_end = cursor, cursor + seg_duration
        cursor = seg_end
        if seg_end <= start_s or seg_start >= end_s:
            continue
        read_from = max(start_s, seg_start) - seg_start
        read_to = min(end_s, seg_end) - seg_start
        first = int(round(read_from * info.samplerate))
        last = int(round(read_to * info.samplerate))
        if last <= first:
            continue
        audio, _ = sf.read(str(path), start=first, stop=last, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def analyze_night(
    session_dir: Path,
    model,
    threshold: float = DEFAULT_THRESHOLD,
    encoder=None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    class_names: list[str] | None = None,
    target_labels: tuple[str, ...] = ("bruxism", "snore"),
    gate_k: float = DEFAULT_GATE_K,
    gate_low_hz: float = DEFAULT_GATE_LOW_HZ,
    gate_high_hz: float = DEFAULT_GATE_HIGH_HZ,
    min_event_s: float = DEFAULT_MIN_EVENT_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    median_frames: int = DEFAULT_MEDIAN_FRAMES,
    probs_cache: Path | None = None,
    on_progress=None,
) -> list[Event]:
    """Detect events across a whole recorded night.

    Args:
        session_dir: A session folder containing ``session.json`` + segments.
        model: The trained head (see ``src.train.train_classifier``).
        threshold: Probability needed to *enter* an event; leaving uses
            ``threshold * HYSTERESIS_EXIT_RATIO``.
        encoder: A ``YamnetEncoder``; created if omitted.
        mean, std: Standardization stats saved beside the model.
        class_names: Class order the model was trained with.
        target_labels: Which classes become events ("other" never does).
        median_frames: Median-filter width in frames; 1 disables smoothing
            (see ``frames_to_events`` for why that is sometimes right).
        probs_cache: Optional .npz holding the per-region class probabilities.
            Everything expensive — streaming the night through the energy gate
            and running YAMNet over each region — depends only on the audio and
            the gate settings, never on the threshold. Caching it turns "what
            would threshold 0.97 have given?" from a 20-minute rerun into a
            couple of seconds. The cache records the gate settings it was built
            with and is ignored if they differ.
        on_progress: Optional ``(stage, done, total)`` callback.

    Returns:
        Events in chronological order, in seconds from the night's start.
    """
    from src.embed.encoder import YamnetEncoder  # noqa: PLC0415
    from src.session.schema import read_session_json  # noqa: PLC0415
    from src.train.train_classifier import predict_proba  # noqa: PLC0415

    session = read_session_json(session_dir)
    if not session.segments:
        return []

    import soundfile as sf  # noqa: PLC0415

    sample_rate = session.sample_rate_actual or sf.info(
        str(session_dir / session.segments[0].file)
    ).samplerate

    encoder = encoder or YamnetEncoder()
    names = list(class_names or ["bruxism", "snore", "other"])
    target_indices = [names.index(label) for label in target_labels if label in names]

    cached = _load_probs_cache(probs_cache, gate_k, gate_low_hz, gate_high_hz, sample_rate)
    if cached is not None:
        regions, probabilities_per_region = cached
        logger.info("ใช้ผลจำแนกที่แคชไว้ %d ช่วง (ข้ามการรัน gate และ YAMNet)", len(regions))
        return _assemble(
            regions, probabilities_per_region, names, target_indices,
            threshold, min_event_s, median_frames, merge_gap_s,
        )

    if on_progress is not None:
        on_progress("gate", 0, 1)
    regions, _total_s = gate_regions(
        session_dir, session, sample_rate, k=gate_k, low_hz=gate_low_hz, high_hz=gate_high_hz
    )

    computed_regions: list[Region] = []
    computed_probs: list[np.ndarray] = []
    encode_failures = 0
    first_failure: Exception | None = None

    for i, region in enumerate(regions, start=1):
        audio = read_window(session_dir, session, region.start_s, region.end_s, sample_rate)
        if audio.size < int(0.1 * sample_rate):
            continue
        try:
            frames = encoder.embed_frames(audio, sample_rate)
        except Exception as error:  # noqa: BLE001 — one bad region shouldn't end the night
            encode_failures += 1
            if first_failure is None:
                first_failure = error
            continue
        if frames.shape[0] == 0:
            continue
        computed_regions.append(region)
        computed_probs.append(predict_proba(model, mean, std, frames))
        if on_progress is not None:
            on_progress("classify", i, len(regions))

    # Tolerating a bad region is right; silently tolerating *every* region is
    # not. A corrupt YAMNet download made embedding fail on all 124 regions of
    # a night and the run still reported "0 events" — indistinguishable from a
    # genuinely quiet night, and the 20 minutes spent on it were wasted.
    if regions and encode_failures == len(regions):
        raise RuntimeError(
            f"ฝัง embedding ล้มเหลวทั้ง {encode_failures} ช่วง — ไม่ใช่คืนที่เงียบ แต่คือโมเดลโหลดไม่ได้"
            f" (ข้อผิดพลาดแรก: {type(first_failure).__name__}: {first_failure})"
        )
    if encode_failures:
        logger.warning(
            "ฝัง embedding ล้มเหลว %d จาก %d ช่วง (ข้ามไป) — ตัวอย่างข้อผิดพลาด: %s",
            encode_failures, len(regions), first_failure,
        )

    _save_probs_cache(
        probs_cache, computed_regions, computed_probs,
        gate_k, gate_low_hz, gate_high_hz, sample_rate,
    )
    return _assemble(
        computed_regions, computed_probs, names, target_indices,
        threshold, min_event_s, median_frames, merge_gap_s,
    )


def _assemble(
    regions, probabilities_per_region, names, target_indices,
    threshold, min_event_s, median_frames, merge_gap_s,
) -> list[Event]:
    """Turn per-region probabilities into merged events.

    Split out so the cached and freshly-computed paths cannot drift apart —
    the whole point of the cache is that the same threshold gives the same
    answer either way.
    """
    events: list[Event] = []
    for region, probabilities in zip(regions, probabilities_per_region):
        events.extend(
            frames_to_events(
                probabilities,
                names,
                region_start_s=region.start_s,
                threshold=threshold,
                target_indices=target_indices,
                min_event_s=min_event_s,
                median_frames=median_frames,
            )
        )
    return merge_events(events, gap_s=merge_gap_s)


def _cache_signature(gate_k, low_hz, high_hz, sample_rate) -> np.ndarray:
    return np.array([gate_k, low_hz, high_hz, float(sample_rate)], dtype=np.float64)


def _load_probs_cache(path, gate_k, low_hz, high_hz, sample_rate):
    if path is None or not Path(path).is_file():
        return None
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if not np.allclose(data["signature"], _cache_signature(gate_k, low_hz, high_hz, sample_rate)):
                logger.warning("แคชถูกสร้างด้วยค่า gate คนละชุด — จะคำนวณใหม่")
                return None
            bounds = data["regions"]
            regions = [Region(float(a), float(b)) for a, b in bounds]
            probs = [data[f"p{i}"] for i in range(len(regions))]
        return regions, probs
    except Exception:  # noqa: BLE001 — a broken cache must never fail the run
        logger.warning("อ่านแคชไม่ได้ — จะคำนวณใหม่")
        return None


def _save_probs_cache(path, regions, probs, gate_k, low_hz, high_hz, sample_rate) -> None:
    if path is None or not regions:
        return
    payload = {f"p{i}": p for i, p in enumerate(probs)}
    payload["regions"] = np.array([[r.start_s, r.end_s] for r in regions], dtype=np.float64)
    payload["signature"] = _cache_signature(gate_k, low_hz, high_hz, sample_rate)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(path), **payload)
    logger.info("บันทึกแคชผลจำแนกไว้ที่ %s", path)
