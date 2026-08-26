# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

BruxEcho — sleep bruxism (tooth-grinding) sound detection from a phone
microphone. Undergraduate AI course project (Computer Engineering). Two
parts: a Python signal-analysis toolkit (`src/`, `tools/`) and an Android
recording app (`android/`). No detection model exists yet — that's a later
sprint.

## Sprint 1 findings (drive most later decisions)

Piloted with a generic phone recorder app and ran `tools/explore.py`
against the clips. Findings:

- The recorder saved **m4a (AAC)** despite a `.wav` extension, and the
  OS/app's voice-call DSP pipeline stripped almost everything above 8 kHz.
- Once high-pass filtered (300 Hz, to remove ambient low-frequency room
  noise that otherwise swamps every energy-ratio number — see README "ทำไม
  ต้อง high-pass filter"), the **8-16 kHz band measured 15.6 dB SNR**
  between grinding and silence — the band that separates the target signal
  from noise best, and exactly the band a voice-tuned pipeline discards.
- Model implication: **YAMNet (16 kHz native) can't see this band; PANNs
  (32 kHz native) can.** `tools/explore.py --input <dir>` prints this
  go/no-go per recording set (see `print_decision_summary`).
- Practical implication: recordings must be raw PCM at ≥44.1 kHz with
  platform audio effects (echo cancellation, noise suppression, AGC)
  disabled, or the 8-16 kHz signal never reaches the file at all. This is
  the entire reason `android/` exists as a native app instead of using any
  off-the-shelf recorder.

## Sprint 2: why the recorder is a native Kotlin app

Considered using a generic recorder app, a cross-platform framework
(Flutter/React Native), or `MediaRecorder`. Rejected all three for the same
reason: none give reliable access to `MediaRecorder.AudioSource.UNPROCESSED`
plus per-effect control (`AcousticEchoCanceler` / `NoiseSuppressor` /
`AutomaticGainControl`) *and* raw PCM output. `MediaRecorder` compresses to
AAC by construction — the Sprint 1 bug. Cross-platform audio plugins
generally wrap `MediaRecorder`, not `AudioRecord`, so the same problem
resurfaces one layer down. Kotlin + `AudioRecord` was the only path that
guarantees PCM16 out and explicit control over which audio source and
effects are active, with the actual source/effects used logged per-session
for the research writeup (`session.json`, `audio_source_used`).

Source fallback chain (first one that reaches `AudioRecord.STATE_INITIALIZED`
wins): `UNPROCESSED` → `CAMCORDER` → `MIC`. Verified on a Samsung Galaxy A55
(SM-A556E, Android 16): `UNPROCESSED` is granted, `AcousticEchoCanceler` and
`NoiseSuppressor` are present and get disabled, `AutomaticGainControl` is
not available on that device at all (not a bug — some OEMs simply don't
expose it; recorded as `effects_unavailable` in `session.json`).

Full technical rationale and Samsung-specific battery/DSP notes are in
[`android/README` section of the top-level README](README.md#แอป-android).

## Sprint 3: energy gate + external-file import

Goal: a first-pass filter that throws out most of a night's audio using
only signal energy (no ML yet), plus a way to test that pipeline against a
night recorded before the app existed, without waiting for the app's own
data to accumulate. Two data sources feed the same downstream tools —
`tools/import_external.py` reshapes an arbitrary external recording into
the same rotating-segment + `session.json` layout the app produces, tagged
`"provenance": "external"` so nothing downstream ever conflates the two
kinds of data (an externally-recorded night's numbers aren't directly
comparable to the app's — different unknown pipeline, no
`audio_source_used`).

**Why band-limited energy, not overall energy** (`src/detect/energy_gate.py`):
Sprint 1's 15.6 dB SNR result was specifically in the 8-16 kHz band; snoring/
HVAC/rumble sit below ~1 kHz and dominate total energy (>90%). Gating on
`[2000, min(16000, 0.95·Nyquist)]` Hz throws out that dominant noise before
any model runs. The high edge adapts per sample rate (and the module raises
if the resulting band is degenerate) so this also works on the external
8-16 kHz-blind file the app was built to fix.

**Why the filter is causal, not zero-phase**: `src/features/spectral.py`'s
`high_pass_filter` uses `sosfiltfilt` (zero-phase) because it runs on short
pilot clips already fully in memory. `energy_gate` streams an 8-hour file
block-by-block instead, so it uses `sosfilt` with filter state (`zi`)
carried across blocks — equivalent to filtering the whole signal at once,
just with a small constant group delay instead of zero phase shift. Fine
for event detection; wrong tool if sub-frame timing precision ever matters.

**Adaptive threshold recomputes ~once/second of audio, not once per frame
or once per caller block** — frame-count-based specifically so a caller's
I/O chunk size never changes detection results (verified by
`test_detection_independent_of_caller_block_size`). Recomputing the
median+MAD from a 5-minute/~12,000-frame rolling window on every 25ms frame
(up to ~1.15M times over 8h) was the difference between the run finishing
in seconds vs. minutes.

**Hysteresis bug caught by the synthetic tests, worth remembering**: the
naive reading of "exit when RMS drops below `threshold * 0.6`" breaks when
`k*MAD` is a small fraction of the median (legitimate for fairly stationary
background noise) — `threshold * 0.6` can then sit *below* the noise floor
itself, so an event, once triggered, never exits. Fixed to a proper
hysteresis band: `median + 0.6 * (threshold - median)`, which is always
between the floor and the entry threshold by construction. This is why
`tests/test_energy_gate.py` exists and uses synthetic (not real) audio —
it caught a real bug before any human ever had to notice a suspiciously
giant "event" in a candidates.csv.

`tools/detect_events.py` streams each session's segments through the gate,
writes `results/events/<session_id>/candidates.csv` + 5s clips + an
`overview.png`, and denormalizes `provenance` onto every CSV row (rather
than requiring `tools/review.py` to relocate the original session
directory, which isn't reliably possible — `--session` can point anywhere).

## Repo layout

- `src/` — pure Python library (audio I/O, spectral feature math, the
  energy gate, the shared `session.json` schema). No matplotlib/CLI/
  streamlit/argparse/side effects; reusable from `tools/`, `tools/review.py`
  (Streamlit), or offline model training.
- `src/session/schema.py` — read/write `session.json`, shared by every tool
  that touches a session so the schema only exists in one place.
- `src/detect/energy_gate.py` — the streaming energy-in-band event gate
  (see Sprint 3 above).
- `tools/explore.py` — CLI that answers the YAMNet-vs-PANNs question from a
  folder of pilot clips: SNR, spectrograms, per-band energy, codec-artifact
  detection.
- `tools/import_external.py` — reshapes one external audio file into a
  `data/raw/sleep/<session_id>/` session (rotating segments + session.json,
  `provenance: external`). Streams the source file; never loads it fully
  into memory. `--dry-run` available; refuses to silently overwrite an
  existing `session_id`.
- `tools/detect_events.py` — runs the energy gate over a session, writes
  `results/events/<session_id>/{candidates.csv, overview.png, clips/}`.
- `tools/review.py` — Streamlit clip-review/labeling tool
  (`streamlit run tools/review.py`); reads only `results/events/`, never
  the raw session directory.
- `android/` — the recording app (Kotlin, Jetpack Compose, `AudioRecord`
  only — never `MediaRecorder`). See its README section for build/install/
  pull commands and the Samsung battery-settings steps.

## Commands

```bash
# Python
uv sync
python -m tools.explore --input data/raw/pilot
python -m tools.import_external --file night.wav --session-id ext_20260820 --dry-run
python -m tools.detect_events --session data/raw/sleep/ext_20260820
streamlit run tools/review.py

# Tests (uv sync --group dev, or: .venv/Scripts/pip install pytest)
pytest

# Android (JDK 17+/21 required; Android Studio's bundled JBR works)
cd android
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ./gradlew.bat :app:installDebug
adb pull /storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions/<id>/ ./data/raw/
```
