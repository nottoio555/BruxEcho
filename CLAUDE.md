# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

BruxEcho — sleep bruxism (tooth-grinding) sound detection from a phone
microphone. Undergraduate AI course project (Computer Engineering). Two
parts: a Python signal-analysis toolkit (`src/`, `tools/`) and an Android
recording app (`android/`). As of Sprint 7 there is a trained classifier
(`models/bruxism_classifier/`) whose output the app reads; see Sprint 7
below for what it can and cannot do yet.

## Sprint 1 findings (drive most later decisions)

Piloted with a generic phone recorder app and ran `tools/explore.py`
against the clips. Findings:

- The recorder saved **m4a (AAC)** despite a `.wav` extension, and the
  OS/app's voice-call DSP pipeline stripped almost everything above 8 kHz.
- Once high-pass filtered (300 Hz, to remove ambient low-frequency room
  noise that otherwise swamps every energy-ratio number — see docs/tools.md
  "ทำไมต้อง high-pass filter"), the **8-16 kHz band measured 15.6 dB SNR**
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

Samsung battery setup is in the [README](README.md#แอป-android); technical
notes are in [`docs/android.md`](docs/android.md).

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

## Sprint 3.5: the gate band is a parameter, not a constant

Sprint 1's 8-16 kHz result came from deliberate tooth-*rubbing* recorded
30 cm from the mic. Clips of real sleep bruxism (`data/raw/grind_clips/
sleep_*`, confirmed by ear) are spectrally almost the opposite: median 0.5%
of energy above 8 kHz (vs 36.7%), centroid 3,051 Hz (vs 9,651 Hz), flatness
0.0013 (vs 0.155), ZCR 0.093 (vs 0.457), and 90% of energy inside
683-5,104 Hz (vs 533-19,496 Hz). Distance plus bedding absorption is the
working hypothesis. The consequence is that the original 2000-16000 Hz band
was gating largely on frequencies the real signal doesn't occupy, which is
the leading explanation for a night yielding 2,536 mostly-noise events.

So `low_hz`/`high_hz` are now caller-supplied end to end (`resolve_band`,
`EnergyGate`, `--low-hz/--high-hz`), and **every `EventCandidate` carries
the `band_low_hz`/`band_high_hz`/`band_mid_hz`/`gate_k` it was produced
with** — a bare candidates.csv otherwise says nothing about how it was made,
which makes two runs impossible to compare after the fact.

`band_ratio_high` was already "upper half of the band / whole band"; what
changed is that the split (`GateBand.mid_hz`) now actually moves, because
`high_hz` moves. With the old fixed 2000-16000 band it was pinned near
9 kHz for every run.

`narrow_band_warning` now means band *width* < 6 kHz (matching the constant's
name and message) rather than "high edge < 6 kHz". Under the old fixed band
those were equivalent; with configurable bands only width is meaningful.

Two tools support choosing the band from data instead of assumption:
`tools/compare_clips.py` (which group's energy sits where) and
`tools/sweep_band.py` (what each band × k would actually have returned).

`detect_events.py` refuses to overwrite an existing `candidates.csv` without
`--force`, because `labels.csv` sits next to it keyed by `event_id` — a
silent rewrite with different settings desyncs hand-labeled data with no
warning.

## Sprint 5: few-shot retrieval, and why the model choice flipped to YAMNet

### YAMNet, not PANNs (this reverses the Sprint 1 decision)

Sprint 1 chose PANNs because the signal lived in 8-16 kHz and YAMNet's
16 kHz sample rate tops out at 8 kHz. That measurement came from deliberate
tooth-*rubbing* at 30 cm. Sprint 3.5 measured real sleep bruxism and found
90% of its energy inside 683-5,104 Hz, median 0.5% above 8 kHz — entirely
under YAMNet's ceiling. So the reason to prefer PANNs no longer holds, and
YAMNet wins on everything else: ~20x smaller, and available as TFLite, so
the same embedding can eventually run inside the Android app instead of
only in this toolkit.

### Why few-shot prototypes instead of training a classifier

There are 6 confirmed sleep-bruxism clips. That cannot train a classifier;
it can define a centroid. `src/embed/prototypes.py` therefore does
nearest-centroid scoring over YAMNet embeddings — nothing is fitted, and
adding a newly confirmed clip is a recomputed mean rather than a retrain.
Cosine similarity specifically, because embedding magnitude tracks loudness
and the reference clips sit ~34 dB above the night's clips.

### Result on ext_night01: the method does not work yet

Measured by `tools/evaluate_similar.py`, with `--crop-to-event`:

- Leave-one-out over the 6 reference clips looks good — median rank 16 of
  2,536, recall@100 = 4/6 — but this is **misleading**: those clips share a
  recording session, so a prototype built from 5 of them partly recognizes
  the session, not the grinding.
- The honest metric is where the night's own 3 human-confirmed grind events
  land: **ranks 16, 449, 1457** (median 449). Clips labelled `other` have
  median rank 567. Grind is barely ahead of arbitrary noise.

So YAMNet embeddings + a 6-clip centroid do **not** separate this night's
grinding. Report it as a negative result rather than quoting the LOO number
alone.

### The clip-length confound (real, partially fixed)

Reference clips are tight crops (0.46-2.31s, nearly all event); candidate
clips are 5s windows where a 0.7s event is ~86% silence. Mean-pooled
embeddings of those differ for reasons unrelated to grinding. Cropping
candidates to their own event window moved the confirmed events from median
rank 1651 to 449 — a real improvement, hence `--crop-to-event` on both
tools, but not enough to make the approach usable.

`src/embed/encoder.py` caches embeddings keyed by file content hash (plus a
transform suffix for cropped variants), because re-encoding 2,536 clips per
run is minutes of GPU-less TensorFlow each time.

## Sprint 7: a trained classifier, and the two metrics that mattered

Sprint 5's negative result was a data problem, not an embedding problem, so
this sprint trained a real head on **DreamCatcher** (event-level sleep-sound
annotations, NeurIPS 2024, CC-BY-4.0). The dataset is **never copied into the
repo** — `configs/paths.yaml` holds the path and everything reads from there.
7 source labels collapse to `bruxism`/`snore`/`other`.

**Split by session, never randomly per clip.** Clips from one night share
mic, room, bed and person; a random split lets the model score well by
recognizing the night. `grouped_split()` keeps a session on one side and
balances the rare bruxism class deterministically. Caveat worth repeating in
any writeup: a session split still shares the same 4 subjects, so
`--group-by subject` is the number to trust for deployment.

**Frozen YAMNet + a 1024→256→64→3 head.** ~5,000 bruxism frames cannot
fine-tune YAMNet without overfitting. 14 min total on CPU, inside the 30-min
budget, so the `--max-other` subsampling path exists but was not needed.

### Frame-level metrics badly understate this model

bruxism frame F1 is 0.308, which looks like failure. But an event spans many
frames and the pipeline needs **one** of them over the threshold, so frame
recall and event recall are different numbers. `tools/evaluate_model.py`
therefore computes both and picks the threshold at event level. This is the
same lesson as Sprint 5's LOO-vs-labelled-ranks: measure two things whose
failure modes differ, or you will confidently report the wrong conclusion.

### "Highest precision" is a trap as a fallback rule

No threshold reaches the 0.80 precision target. The obvious fallback — take
the best precision available — picks 0.995: precision 0.73, but recall 5.9%,
showing 45 of 555 events. For an app whose entire output is "how many times
did you grind", a count that is a tenth of reality is worse than a noisy one.

`choose_event_threshold()` falls back instead to the loosest threshold where
precision has caught up with recall. Since `events_shown / events_real ==
recall / precision`, that is exactly the point where the count is closest to
right: 0.96, showing 506 against a true 555.

### The median filter is off by default (it was deleting true positives)

Two reasons, in order of importance:

1. `evaluate_model` scores an event by the peak of its **raw** frames. A
   median filter in the deployed pipeline would make the shipped detector
   stricter than the evaluated one, so the reported threshold would describe
   a detector that doesn't exist.
2. Measured: the one human-confirmed grind that clears threshold on
   ext_night01 peaks at 0.806 on a *single* frame, with 0.098 and 0.166
   beside it. A 3-frame median returns 0.166 and deletes it. A 3-frame median
   requires two consecutive frames over threshold — ~1.4s of sustained
   grinding — and real bursts are frequently shorter.

YAMNet frames overlap 50% (0.96s window, 0.48s hop), so they are already
smoothed. `--median-frames 3` restores the old behaviour.

### Domain shift is the real open problem

On ext_night01 at the calibrated 0.96: 4 bruxism, 580 snore over 7.87h. The
three human-confirmed grinds score 0.41 (gated out before the model ever
saw it), 0.806, and 0.669 — the model *sees* them but they sit at or under a
threshold calibrated on raw DreamCatcher audio. ext_night01 went through
m4a/AAC and a phone's voice DSP and lost everything above 8 kHz (the Sprint 1
bug), so its probabilities are depressed across the board. One threshold does
not transfer across recording pipelines; the fix is nights recorded by the
app itself (raw PCM, effects off) and a re-calibration.

**A trap to avoid when debugging this:** `read_window()` returns audio at the
file's native rate and ignores the `sample_rate` argument for reading, and
`gate_regions()` uses its `sample_rate` argument for timing. Passing 16000
for a 44.1 kHz session silently reports a 7.87h night as 21.7h and shifts
every timestamp by 2.76x. A diagnostic written that way produced convincing
but entirely wrong per-event scores.

### Leave-one-subject-out: the number for "a new user"

`tools/evaluate_loso.py` holds each of the 4 subjects out in turn, trains on
the rest (early-stopping on sessions carved from the *training* subjects, so
the held-out person cannot leak back in through the stopping rule), and runs
two baselines on identical folds. It never writes to
`models/bruxism_classifier/`. Mean ± std over 4 folds:

| model               | macro F1    | bruxism AP  | ROC-AUC     |
|---------------------|-------------|-------------|-------------|
| MLP head (deployed) | 0.522±0.051 | 0.355±0.053 | 0.905±0.054 |
| Logistic regression | 0.516±0.040 | 0.345±0.076 | 0.906±0.052 |
| Majority class      | 0.250±0.074 | 0.037±0.025 | 0.500       |

Three things to take from it:

1. **Generalization to a stranger barely costs anything at the AP level** —
   0.355 LOSO vs 0.365 on the session split. The subject overlap in the
   session split was not inflating that number much.
2. **The MLP does not beat logistic regression.** Within one std on every
   metric. YAMNet's embedding is doing the work; the two hidden layers are not
   earning their keep. Report it plainly rather than claiming the head matters.
3. **Event recall at the fixed 0.96 falls to 0.24±0.07** (from 0.39). That
   threshold was tuned on the session split; on unseen people the
   probabilities run lower, so fewer events cross it. Event recall at a fixed
   threshold is also not comparable *between* models (logistic regression's
   0.39 reflects differently calibrated probabilities, not a better ranker —
   its AP is lower). Compare models on AP; compare thresholds within a model.

Per-subject AP ranges 0.29-0.43, so a single split can land anywhere in that.
Logistic regression emitted a convergence warning at `max_iter=300`; its
numbers may be very slightly under-fitted.

### The `04_new` filenames and the validation set they skewed

3,000 DreamCatcher files are named `04_new-1-1-user_...`. The session regex
`^(\d+)-(\d+)-(\d+)` did not match them, so nine real sessions collapsed into
one group named `""`. `grouped_split` placed that single group wholly on the
validation side, so it was not a train/val leak — the Sprint 7 numbers stand —
but it made up 55% of validation frames and half its bruxism, and left the
validation set 71% subject 04. The regex now accepts an optional `_tag`: kept
in the session (04_new-1-1 is a different recording from 04-1-1), dropped from
the subject (04_new is subject 04, and splitting by subject must not put one
person on both sides). `tests/test_dataset.py` covers it. The embedding cache
did not need rebuilding — `segment_index` maps each frame back to its segment,
so groups are recomputed from filenames.

### The app reads results, it does not compute them

`RealDetectionRepository` is a pure reader of `events.json` (schema v1;
refuses a higher version rather than guessing). `DetectionRepository` itself
was not modified — `SessionAwareDetectionRepository` is a sub-interface — and
`DetectionRepositoryProvider` routes per call (demo → mock, events.json →
real, neither → mock). The two screens changed one line each.

## Sprint 8: injected ground truth, and two screens that disagreed

### Injecting reference clips into the audio, never into events.json

`tools/inject_clips.py` copies a session, mixes known bruxism clips into the
**audio**, and records where they went in `injected_ground_truth.csv`. The
detector then rediscovers them like any other sound, and
`tools/score_injected.py` reads (never writes) `events.json` to report how
many were found. Writing the events directly would have produced a number
that measured nothing.

Two things that had to be right for the result to mean anything:

- **Place clips only where there is real background** (`active_spans`). 61%
  of the night used here is exact digital silence, an import artifact. Evenly
  spacing across wall-clock time put four of six clips in dead air, where SNR
  against a zero floor is undefined — they went in at unity gain, clipped the
  mix, and reported SNRs of 226 dB. A sound surrounded by pure silence is
  also trivially detectable in a way no real night is.
- **Measure the noise floor locally**, in a 5s window either side. A night's
  floor drifts tens of dB, so one global figure makes some injections
  inaudible and others absurd.

Result on `dc_demo_mixed` at 15 dB SNR: **4 of 6 found** at threshold 0.96.
Report that as recall of the injected clips only — the source is a real sleep
recording that may contain its own grinding, so the other 80 detections are
not automatically false.

### The calendar bug was two bugs, and neither was the one reported

Reported as "two sessions on 18 Sep, calendar shows one, trends shows the
other". Underneath:

1. `daily_summary` is keyed by date but was written once per *session*, so
   the second session of a day silently replaced the first. It is now a true
   per-date aggregate, rebuilt from `sessions`/`events` on every sync — which
   also self-heals rows the old code left wrong, since the per-session loop
   skips sessions that are already synced.
2. The real cause of the reported symptom: **`--push` never sent
   `session.json`**. The calendar reads Room, which is built from
   `session.json`; results and trends read `events.json`. A pushed night had
   only the latter, so it existed on one screen and not the other. `--push`
   now sends both.

Then a third, once the day listed all three sessions: Room was still filled
with **mock** events, so the calendar showed "0 ครั้ง" for a night whose
results screen showed 83. `SessionSync` now reads `events.json` when present
and falls back to mock only for un-analyzed nights. Calendar severity is by
event *duration* (≥2s = severe), not confidence — confidence is how sure the
model is, which is a different question with its own display.

The general shape: any time one fact reaches the user through two code paths,
expect them to disagree, and prefer making one read the other over keeping
them in sync.

### Making a 57-minute file look like a night, honestly

`tools/spread_session.py` builds a new session of a requested length (e.g.
23:30-05:47) by moving the source's active spans to random positions across
it, filling the rest with silence, then the detector is re-run. The
alternative — rewriting timestamps in `events.json` — was rejected twice
over: it invents detections the model never made, and it desynchronizes the
clips, which are cut from real offsets, so an event labelled 03:14 would play
audio from minute 12.

Spans keep their chronological order and only the gaps between them are
randomized. Shuffling would splice unrelated moments together and could
manufacture sounds that were never recorded. `injected_ground_truth.csv` is
remapped along with the audio, so the injected-clip score survives.

One honest caveat on the numbers: spreading raised injected-clip recall from
4/6 to 5/6 and total detections from 84 to 141. That is mostly **not** a
better model — a night that is 94% digital silence gives the energy gate's
`median + k*MAD` threshold a far lower floor, so more audio reaches the
classifier. Don't quote a spread night's counts as if they were a real one's.

### Caching the expensive half so the threshold can be tuned

Everything costly in `analyze_night` — streaming the night through the energy
gate, then YAMNet over each gated region — depends only on the audio and the
gate settings, never on the threshold. `probs_cache` stores the per-region
class probabilities in an `.npz` beside the session, so asking "what would
0.978 have given?" goes from a 20-minute rerun to about a second.
`tools/tune_threshold.py` reads that cache and prints the event count per
threshold; it never writes `events.json`.

Both paths through `analyze_night` call the same `_assemble`, so a cached run
and a fresh one cannot drift apart — which is the only reason the cache is
safe to trust. The cache records the gate settings it was built with and is
ignored if they differ.

Two self-inflicted bugs while adding this, both worth avoiding:

- The patch that introduced the cached branch replaced the **first**
  `return merge_events(events, ...)` in the file, which was the one it had
  just added rather than the one ending the function — deleting the entire
  non-cached path. It failed loudly (`UnboundLocalError`), which is the only
  reason it was caught immediately.
- Splicing the new body in left the old loop orphaned after
  `_save_probs_cache`, where its `on_progress` reference had no binding. The
  cache still wrote before the crash, so the run looked half-successful.

When replacing a block inside a long function, anchor on something unique to
that block, not on a line that repeats.

### Reference clips are a separate screen, deliberately

Since detection finds ~2/3 of known grinding, the app ships the confirmed
clips (`assets/reference_clips/`) so a user can learn the sound. They live on
their own screen, load from assets rather than any session, and carry a
"ไม่ใช่ผลการตรวจจับ" banner — there is no path by which one is counted as an
event.

### Why the clips were inaudible: it was never a volume problem

Clips were peak-normalized to -3 dBFS and still played as a faint thud.
Measuring instead of turning the gain up found why: **95.7% of a clip's
energy sat below 20 Hz**, with DC offsets up to -0.18. Peak normalization
spent the whole headroom budget on inaudible rumble, leaving the grinding
around -45 dBFS in the 300-8000 Hz band a phone speaker can reproduce. The
thud *was* the sub-bass; the grinding was 40+ dB under it.

`src/audio/listening.py` fixes it by working on the band a listener hears:
remove DC, high-pass at 300 Hz (the project's existing cutoff; real bruxism
keeps 90% of its energy in 683-5104 Hz), compress, normalize, limit. Measured
result: **+24.5 dB in the audible band, ~17x louder**, every clip landing at
-20 dBFS with none left below -30.

Two mistakes worth not repeating, both found by measuring rather than
listening:

- **Normalizing on whole-clip RMS overshoots badly.** A 5s window holding a
  1s grind is mostly silence, so its average reads ~-85 dBFS; normalizing
  that to -20 multiplies by 60 dB and throws the grind to +16 dBFS, where the
  limiter claws back 17 dB and the clip ends up quiet again. `active_rms_dbfs`
  takes a high percentile of frame levels so the *burst* is what gets
  measured.
- **Limiting by scaling the whole clip undoes the normalization.** One
  transient sets the peak, everything gets divided by it. The limiter is
  frame-local and downward-only instead. Measured difference between the two:
  10.8 dB vs 24.5 dB of gain.

Order matters too: the pre-gain has to come *before* compression, because the
compressor's floor is an absolute dBFS value and raw night audio near
-85 dBFS otherwise counts entirely as "too quiet to touch".

`--clips-only` re-cuts the listening copies from an existing `events.json`
without re-running detection, because tuning how a clip sounds has nothing to
do with detection and a re-run costs ~20 minutes per attempt. It never
rewrites `events.json`.

### The hiss fix: three attempts, and the dull one won

Making the clips audible made the room's hiss audible too ("เสียงซ่าดังมาก").
Scored across 50 real clips — grind-to-background contrast, and the level of
the 5-8 kHz band the hiss lives in:

| chain                    | contrast | hiss  |
|--------------------------|----------|-------|
| level only               | 12.4 dB  | +10.0 |
| + band-limit 400-5000 Hz | 12.2 dB  |  -5.4 |
| + expansion              |  8.5 dB  |  -8.1 |
| + compression            |  3.6 dB  | +11.5 |

- **Compression** (the first attempt) was the worst thing available. Lifting
  quiet frames toward a target lifts the hiss with them: contrast fell from
  12.4 to 3.6 dB, which is why the clips came back as mostly hiss.
- **Expansion** was the obvious correction and measured *worse than doing
  nothing* — 2.7 dB less hiss for 3.7 dB less contrast, because the gain
  reduction smears into the grind's own attack and decay.
- **Band-limiting** — keep 400-5000 Hz, throw the rest away — cut the hiss
  15.4 dB at a cost of 0.2 dB of contrast. Grinding peaks at 700-3000 Hz and
  is ~9 dB down by 5 kHz while hiss is broadband, so the top octave is nearly
  all hiss.

Both dynamics stages are therefore **off by default** and remain available.

**Level is set by peak, not RMS.** Peak normalization failed originally only
because the peak was a DC-shifted subsonic rumble; once that is gone the peak
*is* the grind transient. Crest factor is ~40 dB, so any RMS target loud
enough to hear forces the limiter to flatten exactly the transients that
carry the information. Measured sweep of extra gain above peak-normalized:

| extra | contrast | level      | hiss  |
|-------|----------|------------|-------|
| +0 dB | 12.2 dB  | -40.9 dBFS | -26.3 |
| +12 dB| 10.0 dB  | -29.0 dBFS | -14.3 |
| +24 dB|  5.3 dB  | -18.5 dBFS |  -3.3 |

Every 6 dB of loudness costs 1-2.5 dB of contrast and adds 6 dB of hiss.
`DEFAULT_EXTRA_GAIN_DB = 12.0` is the chosen point and is the knob to turn.

The generalizable lesson: for this kind of complaint, measure a
signal-to-background ratio and the offending band separately. "Make it
louder" and "make it clearer" pull in opposite directions, and the intuitive
dynamics tools made it measurably worse both times.

### Clips are normalized when written, not when pushed

`process_session` normalizes each `analysis_clips/` file to -3 dBFS peak. A
night's background sits near -45 dBFS, so raw 5s windows were close to
inaudible on a phone speaker. Safe because nothing analyses those files — the
detector reads the original segments.

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
  `--low-hz/--high-hz` pick the gate band; refuses to overwrite an existing
  candidates.csv without `--force` (see Sprint 3.5 above).
- `tools/sweep_band.py` — runs many band × k combinations over one session
  and reports counts only (no clips, no per-run charts) so a 15-config sweep
  over an 8-hour night takes ~17 min. Writes
  `results/sweep/<session_id>/{sweep_results.csv, heatmap_events.png}`.
- `tools/compare_clips.py` — feature comparison between two groups of clips
  split by filename prefix (`awake_` vs `sleep_`). Writes
  `results/compare_clips/{clip_features.csv, group_summary.csv,
  boxplots.png, band_profile.png}`.
- `src/embed/encoder.py` — YAMNet embeddings + content-hash disk cache.
  TensorFlow is imported lazily so importing the module stays cheap.
- `src/embed/prototypes.py` — nearest-centroid scoring (cosine). No training.
- `src/train/dataset.py` — DreamCatcher TSV -> 3 classes + grouped split.
- `src/train/train_classifier.py` — frame embeddings, bulk npz cache, head.
- `src/detect/pipeline.py` — whole-night analysis to `list[Event]`.
- `src/audio/mixing.py` — trim/fade/gain/placement math for injecting
  reference clips at a controlled SNR. Pure functions, no I/O.
- `src/audio/listening.py` — makes a clip audible to a person (DC removal,
  high-pass, compression, audible-band normalization, limiting).
  Presentation only; the detector never reads these files.
- `src/embed/references.py` — assembles prototype reference clips from
  `labels.csv` + `candidates.csv` + `data/raw/grind_clips`. Uses stdlib `csv`
  (not pandas) to keep `src/` off the analysis stack. Excludes `awake_*`
  clips from the grind prototype and drops the "unsure" label.
- `tools/detect_similar.py` — ranks a night's clips by similarity to the
  grind prototype; writes `scored.csv`, `prototypes.csv`, and amplified
  `ranked_clips/` in listening order. Use `--crop-to-event`.
- `tools/evaluate_similar.py` — leave-one-out ranking test + where the
  night's own labelled events land + score distribution by human label.
- `tools/evaluate_loso.py` — leave-one-subject-out with majority-class and
  logistic-regression baselines; read-only against the deployed model.
- `tools/train_model.py` / `tools/evaluate_model.py` — train the head and
  pick the deployment threshold (event level; writes it into the model's
  metadata.json so evaluating and deploying can't drift apart).
- `tools/process_session.py` — analyze one night -> `events.json` + clips,
  `--push` sends session.json + events.json + the playable clips to the
  phone with adb.
- `tools/inject_clips.py` — mix known clips into a copy of a session's
  audio at a target SNR; writes `injected_ground_truth.csv`.
- `tools/score_injected.py` — read-only comparison of `events.json`
  against that ground truth.
- `tools/set_session_time.py` — fix an imported session's `started_at`
  without re-importing its audio.
- `tools/spread_session.py` — stretch a short recording across a full
  night by randomizing the silence between its audio, then re-detect.
- `tools/tune_threshold.py` — read-only: event count per threshold, from
  the cached probabilities, without re-running detection.
- `tools/review.py` — Streamlit clip-review/labeling tool
  (`streamlit run tools/review.py`); reads only `results/events/`, never
  the raw session directory.
- `android/` — the recording app (Kotlin, Jetpack Compose, `AudioRecord`
  only — never `MediaRecorder`). See its README section for build/install/
  pull commands and the Samsung battery-settings steps.
- `docs/` — Thai reader docs split out of the README: `tools.md` (CLI manual),
  `development-log.md` (per-sprint history), `android.md`,
  `recording-protocol.md`. The README stays a short overview for the course
  advisor; put new detail here, not there.

## Commands

```bash
# Python
uv sync
python -m tools.explore --input data/raw/pilot
python -m tools.import_external --file night.wav --session-id ext_20260820 --dry-run
python -m tools.detect_events --session data/raw/sleep/ext_20260820
python -m tools.detect_events --session <path> --low-hz 1000 --high-hz 6000 --output results/events_1k6k
python -m tools.compare_clips --dir data/raw/grind_clips
python -m tools.sweep_band --session data/raw/sleep/ext_night01 --max-hours 0.5
streamlit run tools/review.py

# Sprint 5 — few-shot similarity ranking (needs tensorflow, ~500 MB)
python -m tools.detect_similar --candidates results/events/ext_night01/candidates.csv \
    --references data/raw/grind_clips --labels results/events/ext_night01/labels.csv \
    --crop-to-event --output results/similar/ext_night01
python -m tools.evaluate_similar --candidates results/events/ext_night01/candidates.csv \
    --references data/raw/grind_clips --labels results/events/ext_night01/labels.csv \
    --crop-to-event --output results/similar/ext_night01/eval

# Tests (uv sync --group dev, or: .venv/Scripts/pip install pytest)
pytest

# Android (JDK 17+/21 required; Android Studio's bundled JBR works)
cd android
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ./gradlew.bat :app:installDebug
adb pull /storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions/<id>/ ./data/raw/
```
