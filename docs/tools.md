# คู่มือเครื่องมือ

รายละเอียดการใช้เครื่องมือบรรทัดคำสั่งใน `tools/` ขั้นตอนหลักสำหรับวิเคราะห์หนึ่งคืน
อยู่ใน [README](../README.md#คู่มือการใช้งาน) ส่วนเหตุผลและผลทดลองของแต่ละ Sprint
อยู่ใน [บันทึกการพัฒนา](development-log.md)

ทุกคำสั่งรันจากโฟลเดอร์รากของโปรเจกต์

## รายการเครื่องมือ

| กลุ่ม | เครื่องมือ | หน้าที่ |
|---|---|---|
| วิเคราะห์หนึ่งคืน | `tools/process_session.py` | รันทั้ง pipeline เขียน `events.json` และคลิป, `--push` ส่งเข้ามือถือ |
| | `tools/tune_threshold.py` | ดูจำนวน event ต่อ threshold จากแคช โดยไม่แก้ `events.json` |
| เทรน/ประเมินโมเดล | `tools/train_model.py` | เทรนหัวจำแนกบน DreamCatcher |
| | `tools/evaluate_model.py` | ประเมินระดับเฟรมและ event แล้วบันทึก threshold ลง metadata ของโมเดล |
| | `tools/evaluate_loso.py` | leave-one-subject-out พร้อม baseline (ไม่แตะโมเดลที่ใช้งาน) |
| ทดสอบด้วยเสียงที่รู้คำตอบ | `tools/inject_clips.py` | ผสมคลิปที่ยืนยันแล้วลงในสำเนาของ session |
| | `tools/score_injected.py` | เทียบ `events.json` กับตำแหน่งที่แทรก |
| | `tools/spread_session.py` | ยืดไฟล์สั้นให้เต็มคืนด้วยการสุ่มช่วงเงียบ แล้วตรวจจับใหม่ |
| | `tools/build_dreamcatcher_night.py` | ประกอบคืนเต็มจากคลิป DreamCatcher เพื่อใช้เป็น session สาธิต |
| จัดการ session | `tools/import_external.py` | นำเข้าไฟล์เสียงจากภายนอกให้อยู่ในรูปแบบ session |
| | `tools/set_session_time.py` | แก้ `started_at` โดยไม่ต้อง import ใหม่ |
| สำรวจสัญญาณ | `tools/explore.py` | SNR ต่อย่านความถี่, spectrogram, ตรวจ codec artifact |
| | `tools/compare_clips.py` | เทียบฟีเจอร์เสียงระหว่างสองกลุ่มคลิป |
| | `tools/sweep_band.py` | กวาดย่านความถี่ × k แล้วรายงานจำนวน event |
| | `tools/detect_events.py` | รัน energy gate อย่างเดียว ตัดคลิปสำหรับฟัง |
| | `tools/find_new_events.py` | หา event ที่ย่านใหม่เจอแต่ย่านเดิมพลาด |
| ตรวจทานด้วยหู | `tools/review.py` | หน้า Streamlit สำหรับฟังและติด label |
| | `tools/listen.py` | ดึงคลิปที่ยาวที่สุดมาขยายเสียงและเรียงลำดับ |
| | `tools/boost_clips.py` | ตรวจและขยายระดับเสียงของคลิป |
| งานวิจัยที่ไม่ได้ใช้ใน pipeline หลัก | `tools/detect_similar.py`, `tools/evaluate_similar.py` | การจัดอันดับแบบ few-shot จาก [Sprint 5](development-log.md#sprint-5-ตรวจจับด้วยความคล้าย-few-shot--และผลที่ได้จริง) ซึ่งเป็นผลลบ |
| แอป | `tools/regenerate_brand_assets.py` | สร้างไอคอนแอปจาก `brand/logo.png` |

## วิเคราะห์หนึ่งคืน — `tools/process_session.py`

```bash
python -m tools.process_session --session data/raw/sleep/<session_id>
python -m tools.process_session --session <path> --threshold 0.9      # ไม่ใส่ = ใช้ค่าจาก metadata ของโมเดล
python -m tools.process_session --session <path> --push               # ส่ง session.json + events.json + คลิปเข้ามือถือ
python -m tools.process_session --session <path> --clips-only --push  # ตัดคลิปใหม่จาก events.json เดิม
python -m tools.tune_threshold --session <path>                       # ดูจำนวน event ต่อ threshold โดยไม่รันใหม่
```

ออปชันที่ใช้บ่อย: `--gate-k` (ค่าเริ่มต้น 6), `--median-frames` (ค่าเริ่มต้น 1 = ปิด),
`--device` (ระบุเครื่องเมื่อต่อ adb หลายเครื่อง)

## เทรนและประเมินโมเดล

ต้องมีชุดข้อมูล DreamCatcher และตั้ง path ใน `configs/paths.yaml` ก่อน

```bash
python -m tools.train_model --group-by session
python -m tools.evaluate_model
python -m tools.evaluate_loso
```

## ทดสอบด้วยเสียงที่รู้คำตอบ

```bash
python -m tools.inject_clips --session data/raw/sleep/dc_demo \
    --clips "data/raw/grind_clips/sleep_sister_0*.wav" \
    --snr-db 15 --output data/raw/sleep/dc_demo_mixed
python -m tools.process_session --session data/raw/sleep/dc_demo_mixed
python -m tools.score_injected --session data/raw/sleep/dc_demo_mixed
```

## สำรวจสัญญาณ — `tools/explore.py`

```bash
python -m tools.explore --input data/raw/pilot
python -m tools.explore --input data/raw/pilot --baseline 01_silence.wav
python -m tools.explore --input data/raw/pilot --output results/explore/run2
python -m tools.explore --input data/raw/pilot --highpass 0     # ปิด high-pass filter
python -m tools.explore --input data/raw/pilot --trim-start 1.0 # ตัดต้นคลิปนานกว่า default
```

### ทำไมต้อง high-pass filter

รอบทดลองแรกพบว่าย่าน **0-500 Hz กินพลังงานรวมของทุกคลิปถึง ~90%**
(รวมคลิปเงียบด้วย) เพราะสภาพแวดล้อมอัดมีเสียงครางความถี่ต่ำ (เช่น
มอเตอร์/โครงสร้างอาคาร) ปนอยู่ตลอดเวลา พลังงานก้อนนี้ไม่เกี่ยวกับเสียง
กัดฟันเลย แต่มันครอบงำตัวเลข "สัดส่วนพลังงานทั้งหมด" จนทำให้
`high_freq_ratio_pct` ต่ำผิดปกติ (0.47% สำหรับคลิปถูฟัน) — ทั้งที่ดูจาก
spectrogram แล้วเห็นพลังงานจากการถู/เคาะฟันชัดเจนไปถึง 20 kHz

`explore.py` จึงรัน **Butterworth high-pass filter (order 4, zero-phase)**
ที่ 300 Hz เป็นค่าเริ่มต้นก่อนคำนวณทุกอย่าง (ปรับผ่าน `--highpass`, ใส่ 0
เพื่อปิด) — ยกเว้น `01_spectrograms.png` ที่จงใจวาดทั้งสองแบบ (raw กับ
filtered) คู่กันไว้เทียบ เพื่อยืนยันด้วยตาว่า filter ไม่ได้ตัดสัญญาณจริง
ทิ้งไปด้วย

## วิธีอ่านผลลัพธ์ของ explore.py

ทุกอย่างถูกบันทึกไว้ใน `results/explore/` (ไม่ถูก track โดย git):

- **`summary.csv`** — ตารางค่าที่คำนวณได้ทั้งหมดต่อไฟล์ (RMS, peak, SNR รวม,
  **SNR แยกย่านความถี่** — `snr_300_2k_db`, `snr_2k_4k_db`, `snr_4k_8k_db`,
  `snr_8k_16k_db`, `snr_16k_plus_db`, สัดส่วนพลังงานตามย่านความถี่,
  spectral centroid/rolloff/flatness, ZCR) — ทุกค่านี้คำนวณจากสัญญาณ
  ที่ผ่าน high-pass filter แล้ว
- **`01_spectrograms.png`** — log-mel spectrogram ของทุกไฟล์ **สองคอลัมน์
  (raw | high-pass)** เทียบกัน มีเส้นแดงประที่ 8 kHz กำกับว่า "YAMNet limit"
- **`02_psd_comparison.png`** — Power Spectral Density ของทุกไฟล์ (หลัง
  high-pass) ซ้อนกัน แกน x เป็น log scale เทียบรูปทรงสเปกตรัมระหว่างไฟล์
  ได้ง่าย
- **`03_band_energy.png`** — stacked bar แสดงสัดส่วนพลังงานแต่ละย่าน
  ความถี่ต่อไฟล์ (หลัง high-pass)
- **`04_waveform_envelope.png`** — waveform + RMS envelope ตามเวลา (หลัง
  high-pass) ดูรูปแบบการถู (ต่อเนื่อง) เทียบกับการเคาะ (เป็นจังหวะ)
- **`05_band_snr.png`** — grouped bar chart ของ SNR แต่ละย่านความถี่
  (เทียบ baseline) ต่อคลิป มีเส้นแดงประที่ 6 dB เป็นเกณฑ์ "แยกได้"

### เกณฑ์การเลือกโมเดล

ใช้ **SNR แยกย่านความถี่** แทน `high_freq_ratio_pct` เพราะสัดส่วนพลังงาน
ทั้งหมดถูกกดต่ำผิดปกติได้ง่ายเมื่อย่านความถี่ต่ำมีพลังงานเยอะ (ปัญหาข้างต้น)
ในขณะที่ SNR ต่อย่านบอกตรงๆ ว่าย่านนั้น "แยกจาก noise floor ได้แค่ไหน"
ไม่ว่าย่านอื่นจะมีพลังงานเท่าไหร่ก็ตาม — console จะพิมพ์ SNR เฉลี่ยของทุกย่าน
(จากคลิปถูฟัน) แล้วสรุป:

- ถ้า **`snr_8k_16k_db` เฉลี่ย ≥ 6 dB** → ย่านเหนือ 8 kHz มีสัญญาณจริง
  แนะนำใช้ **PANNs (32 kHz)** แทน YAMNet (16 kHz) หรือเสริมฟีเจอร์
  ย่านความถี่สูงเข้าไป
- ถ้า **< 6 dB** → YAMNet (16 kHz) น่าจะเพียงพอ

ผลลัพธ์ข้อนี้ใช้เป็นเหตุผลประกอบการเลือกโมเดลในรายงาน

### การตรวจจับ codec/upsample artifact

เครื่องมือจะสแกนหาการลดพลังงานฉับพลัน (>10 dB ในช่วงความถี่แคบ) ที่เกิดขึ้น
**ที่ความถี่เดียวกันในทุกไฟล์** (รันบนสัญญาณ raw ก่อน high-pass) — ถ้าเจอ
จะพิมพ์ WARNING เตือนว่าอาจเป็นร่องรอยของการบีบอัด/อัปแซมเปิลเสียงในสาย
การอัด/แปลงไฟล์ และตัวเลขย่านความถี่สูงอาจไม่น่าเชื่อถือ 100%

## energy gate และการนำเข้าไฟล์ภายนอก

ก่อนข้อมูลจากแอป BruxEcho เองจะสะสมพอ ระบบรองรับการทดสอบไปป์ไลน์ด้วยไฟล์
เสียงที่อัดจากแอปอื่นได้ด้วย — ทั้งสองแหล่งไหลผ่านเครื่องมือชุดเดียวกัน
โดยแยกกันด้วยฟิลด์ `provenance` ใน session.json (`bruxecho_app` หรือ
`external`) เพื่อไม่ให้เอาตัวเลขจากสองแหล่งมาเทียบกันตรงๆ (แหล่งภายนอกไม่รู้
audio pipeline ที่แท้จริง)

### 1. นำเข้าไฟล์ภายนอก — `tools/import_external.py`

```bash
python -m tools.import_external --file "path/to/night.wav" --session-id ext_20260820 --dry-run   # ดูแผนก่อน ไม่แตะไฟล์จริง
python -m tools.import_external --file "path/to/night.wav" --session-id ext_20260820              # นำเข้าจริง
```

อ่านไฟล์ต้นฉบับแบบ **streaming** (ไม่โหลดทั้งไฟล์เข้า RAM แม้จะยาว 8
ชั่วโมง) แบ่งเป็น segment ละ ~1 ชั่วโมงแบบเดียวกับแอป, ดาวน์มิกซ์ stereo
เป็น mono, และสร้าง `session.json` ให้อัตโนมัติ พร้อมเตือนชัดเจนถ้า:

- ไฟล์ไม่ใช่ PCM (เช่น ADPCM/μ-law หรือถูกบีบอัดมาก่อน) — ข้อมูลความถี่สูง
  อาจสูญหายไปแล้ว ตัวเลขที่ได้ใช้อ้างอิงในรายงานไม่ได้เต็มร้อย
- sample rate ต่ำกว่า 32 kHz — ย่าน 8-16 kHz วิเคราะห์ไม่ได้
- เปิดไฟล์ด้วย soundfile ไม่ได้เลย (เช่น เป็น AAC/M4A ที่เปลี่ยนนามสกุลเป็น
  .wav — ปัญหาเดิมจาก Sprint 1) จะบอกคำสั่ง `ffmpeg` สำหรับแปลงไฟล์ให้

**ความปลอดภัยของข้อมูล**: คัดลอกไฟล์ต้นฉบับ (byte-exact) เก็บไว้ที่
`session_dir/source/` เสมอ ไม่ย้าย/ลบ/เขียนทับไฟล์ต้นฉบับเด็ดขาด และถ้า
`session_id` ซ้ำกับที่มีอยู่แล้ว จะหยุดถามก่อนทุกครั้ง (ไม่เขียนทับเงียบๆ)

### 2. ตัวกรองด่านแรกด้วยพลังงาน — `src/detect/energy_gate.py`

กรองด้วย**พลังงานย่านความถี่** (ค่าเริ่มต้น [2000, min(16000, 0.95×Nyquist)] Hz
ปรับได้ด้วย `--low-hz/--high-hz` — ค่านี้เป็นของ `tools/detect_events.py` ส่วน pipeline หลักใช้ 800-4000 Hz ดูหัวข้อ "ปรับย่านความถี่" ด้านล่าง)
ไม่ใช่พลังงานรวม เพราะเสียงกรน/แอร์/รัมเบิลอยู่ต่ำกว่า 1 kHz และกินพลังงาน
รวมกว่า 90% (ผล Sprint 1) ในขณะที่เสียงกัดฟันเด่นชัดในย่านสูง (SNR 15.6 dB)
เกณฑ์ปรับตัวได้จาก `median + k×MAD` ของหน้าต่างเลื่อน 5 นาทีย้อนหลัง พร้อม
hysteresis กันสัญญาณกระพริบที่ขอบ threshold — เป็นไลบรารีล้วน (`src/`)
สตรีมทีละ block จริง ไม่โหลดทั้งคืนเข้า RAM ดูเหตุผลเชิงเทคนิคเต็มๆ
(ทำไมใช้ causal filter, ทำไม threshold คำนวณใหม่ทุก ~1 วินาทีไม่ใช่ทุกเฟรม,
บั๊ก hysteresis ที่ tests จับได้) ใน [`CLAUDE.md`](../CLAUDE.md)

### 3. รัน detection — `tools/detect_events.py`

```bash
python -m tools.detect_events --session data/raw/sleep/ext_20260820
python -m tools.detect_events --session <path> --k 5 --min-duration 0.3
python -m tools.detect_events --session <path> --low-hz 1000 --high-hz 6000 \
    --output results/events_1k6k        # ทดลองย่านอื่น แยกโฟลเดอร์ผลลัพธ์
```

สร้างที่ `results/events/<session_id>/`:

- **`candidates.csv`** — event ที่ตรวจจับได้ทั้งหมด พร้อมเวลา, RMS,
  SNR เหนือ noise floor, สัดส่วนพลังงานครึ่งบนของย่าน (`band_ratio_high`)
  และ**ค่าที่ใช้ตรวจจับจริงของแถวนั้น** (`band_low_hz`, `band_high_hz`,
  `band_mid_hz`, `gate_k`) เพื่อให้เทียบผลข้ามการรันได้โดยไม่ต้องเดา

> ถ้ามี `candidates.csv` อยู่แล้วในโฟลเดอร์ปลายทาง เครื่องมือจะ**ปฏิเสธ
> การเขียนทับ** (ต้องใส่ `--force`) เพราะ `labels.csv` ที่ `tools/review.py`
> สร้างไว้อ้าง `event_id` เดิม — เขียนทับเงียบๆ แล้ว label จะไม่ตรงกับ event
> อีกต่อไป เวลาทดลองชุดค่าใหม่ให้ใช้ `--output` แยกโฟลเดอร์เสมอ
- **`clips/`** — คลิป ~5 วินาทีต่อ event (event ± padding) สำหรับฟังจริง
- **`overview.png`** — กราฟ RMS ทั้งคืน (แกน x เป็นเวลานาฬิกาจริง) พร้อม
  เส้น adaptive threshold และจุดแดงตรงตำแหน่ง event

พิมพ์สรุปจำนวน event, event/ชั่วโมง, การกระจายรายชั่วโมง, สัดส่วนของคืนที่
ถูกคัดออก, ย่านความถี่ที่ใช้จริง — และเตือนพร้อมแนะนำค่า `--k` ใหม่ถ้าได้
event มากกว่า 2000 หรือน้อยกว่า 20 ครั้ง (เป้าหมาย 100-500 ครั้ง/คืน)

### 4. ตรวจทาน/ติด label — `tools/review.py`

```bash
streamlit run tools/review.py
```

เครื่องมือฟังและติด label — เลือก session
(แสดง provenance กำกับ), เรียงคลิปตาม peak RMS มากไปน้อยเป็นค่าเริ่มต้น
(เจอของจริงเร็วสุด), มี spectrogram + ปุ่มขยายเสียง 1x/5x/20x (เสียงจริง
เบามาก), ปุ่ม label กัดฟัน/กรน/อื่นๆ/ไม่แน่ใจ บันทึกลง `labels.csv` ทันที
ทุกครั้งที่กด กลับมาทำต่อจากที่ค้างได้ อ่านเฉพาะจาก `results/events/` —
ไม่แตะไฟล์ session ต้นฉบับเลย

## ปรับย่านความถี่ให้ตรงกับเสียงกัดฟันจริง

**ปัญหาที่เจอ:** ย่าน 2000-16000 Hz มาจากผล Sprint 1 ซึ่งวัดจากเสียง**ถูฟัน
ตอนตื่น** ห่างไมค์ 30 ซม. แต่พอเอาคลิปเสียง**กัดฟันจริงตอนหลับ** (ยืนยันด้วย
การฟังแล้ว) มาวัด กลับได้สเปกตรัมคนละแบบ:

| | ถูฟัน 30 ซม. (`awake_*`) | กัดฟันจริงตอนหลับ (`sleep_*`) |
|---|---|---|
| พลังงาน >8 kHz | 36.7% | **0.5%** |
| spectral centroid | 9,651 Hz | **3,051 Hz** |
| spectral flatness | 0.155 | **0.0013** |
| ZCR | 0.457 | **0.093** |
| พลังงาน 90% กลางอยู่ที่ | 533-19,496 Hz | **683-5,104 Hz** |

(ตัวเลข median จาก `tools/compare_clips.py` — ระยะห่างและการดูดซับของ
เครื่องนอนน่าจะกินย่านความถี่สูงไป) แปลว่าย่านเดิม**กรองอยู่ในที่ที่สัญญาณจริง
แทบไม่มี** ซึ่งน่าจะเป็นสาเหตุที่รันคืนหนึ่งแล้วได้ 2,536 event ที่ส่วนใหญ่
เป็นเสียงรบกวน

### 5. เทียบฟีเจอร์เสียงระหว่างสองกลุ่ม — `tools/compare_clips.py`

```bash
python -m tools.compare_clips --dir data/raw/grind_clips
```

แยกกลุ่มจาก prefix ของชื่อไฟล์ (`awake_*` vs `sleep_*`) แล้วคำนวณ duration,
RMS, สัดส่วนพลังงาน 10 ย่าน, centroid, flatness, ZCR, rolloff และความถี่ที่
พลังงานสะสมถึง 5/50/95% ออกที่ `results/compare_clips/`:
`clip_features.csv`, `group_summary.csv`, `boxplots.png` (box plot + จุดของ
ทุกคลิป เพราะกลุ่มละ 5-6 คลิป กล่องอย่างเดียวซ่อนว่าความต่างมาจากคลิปเดียว
หรือไม่) และ `band_profile.png` (แท่งเทียบว่าพลังงานของแต่ละกลุ่มอยู่ย่านไหน)

console จะเตือนด้วยว่า RMS/peak **เทียบข้ามกลุ่มไม่ได้** ถ้าคลิปมาจากคนละ
ไฟล์ต้นทาง/คนละ gain (ในชุดนี้ต่างกันถึง ~34 dB) — ให้ดูเฉพาะฟีเจอร์ที่ไม่
ขึ้นกับความดัง

### 6. กวาดหาย่านที่เหมาะ — `tools/sweep_band.py`

```bash
python -m tools.sweep_band --session data/raw/sleep/ext_night01
python -m tools.sweep_band --session <path> --max-hours 0.5        # ลองเร็วๆ ก่อน
python -m tools.sweep_band --session <path> --bands 500-4000 1000-6000 --k 6 12
```

รันย่าน × k หลายชุดบนคืนเดียวกัน (ค่าเริ่มต้น 5 ย่าน × k 6/8/12 = 15 ชุด)
แล้วสรุปเป็นตาราง band | k | จำนวน event | event/ชม. | % ของคืนที่ถูกคัดออก |
median ความยาว event — **ไม่ตัดคลิปและไม่วาดกราฟรายชุด** เพื่อให้เร็วพอ
กวาดทั้งคืนได้ (~17 นาทีสำหรับคืน 7.9 ชั่วโมง) ผลออกที่
`results/sweep/<session_id>/`: `sweep_results.csv` + `heatmap_events.png`

ใช้ย่านที่ดูดีแล้วค่อยกลับไปรัน `detect_events.py` ด้วย `--low-hz/--high-hz`
(และ `--output` แยก) เพื่อตัดคลิปมาฟังจริง — **จำนวน event ที่ดูสวยไม่ได้แปลว่า
ย่านนั้นถูกต้อง** ต้องฟังยืนยันเสมอ
