# BruxEcho

ระบบตรวจจับเสียงการนอนกัดฟัน (sleep bruxism) จากไมค์มือถือ — อัดเสียงทั้งคืน
แล้วใช้ deep learning แยกเสียงกัดฟันออกจากเสียงกรน/พลิกตัว/เสียงรบกวน
เพื่อรายงานว่ากัดกี่ครั้ง เมื่อไหร่ และนานเท่าไหร่

โปรเจควิชา AI ระดับปริญญาตรี สาขาวิศวกรรมคอมพิวเตอร์

**สถานะปัจจุบัน:** Sprint 3 — มีแอป Android สำหรับอัดเสียงทั้งคืน
(`android/`), เครื่องมือสำรวจสัญญาณเสียง (`tools/explore.py`), ตัวกรองด่านแรก
ด้วยพลังงานย่านความถี่ (`src/detect/energy_gate.py` + `tools/detect_events.py`)
และเครื่องมือนำเข้า/ตรวจทานไฟล์เสียง (`tools/import_external.py`,
`tools/review.py`) ยังไม่มีโมเดล ML ตรวจจับเสียงกัดฟันโดยตรง

## ติดตั้ง

ใช้ [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

หรือถ้าไม่มี uv ใช้ venv ธรรมดา:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install numpy scipy librosa soundfile matplotlib pandas streamlit
pip install pytest           # ถ้าต้องการรัน test (pytest)
```

## โปรโตคอลการอัดเสียงชุดทดลอง (pilot)

อัดในห้องเงียบ **ปิดแอร์/พัดลม** ไมค์ (มือถือ) วางห่างจากปาก **30 ซม.**
บันทึกเป็น **WAV, 44.1 kHz** ความยาวคลิปละ **~10 วินาที** วางไฟล์ไว้ที่
`data/raw/pilot/` โดยตั้งชื่อไฟล์ตามรูปแบบนี้ (ตัวเลขนำหน้าช่วยเรียงลำดับ):

| ไฟล์ | คำอธิบาย |
|---|---|
| `01_silence.wav` | ความเงียบ — ใช้เป็น baseline วัด SNR |
| `02_grind_wet.wav` | ถูฟันปกติ (ปากเปียก) |
| `03_grind_dry.wav` | ถูฟันหลังซับปากแห้ง |
| `04_grind_lateral.wav` | เลื่อนกรามออกด้านข้าง |
| `05_grind_protrusive.wav` | เลื่อนกรามไปหน้า-หลัง |
| `06_tapping.wav` | เคาะฟันกระทบกัน |
| `07_grind_1m.wav` | ถูฟันโดยวางไมค์ห่าง 1 เมตร |

ไม่จำเป็นต้องอัดครบทุกไฟล์ — `explore.py` ทำงานได้กับไฟล์เท่าที่มี
ไฟล์ baseline จะถูกจับจากชื่อไฟล์ที่มีคำว่า `silence` โดยอัตโนมัติ
(หรือระบุเองผ่าน `--baseline`)

**สำคัญ:** ต้องอัดที่ 44.1 kHz (หรือสูงกว่า) เพื่อให้ตอบคำถามเรื่องพลังงาน
เหนือ 8 kHz ได้ — ถ้าอัดที่ 16 kHz หรือต่ำกว่า เครื่องมือจะเตือนว่าผลลัพธ์
ในย่านความถี่สูงเชื่อถือไม่ได้

## วิธีรัน explore.py

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

## วิธีอ่านผลลัพธ์

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

## Sprint 3: ตัวกรองด่านแรก (energy gate) + นำเข้าไฟล์ภายนอก

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

กรองด้วย**พลังงานย่านความถี่สูง** ([2000, min(16000, 0.95×Nyquist)] Hz)
ไม่ใช่พลังงานรวม เพราะเสียงกรน/แอร์/รัมเบิลอยู่ต่ำกว่า 1 kHz และกินพลังงาน
รวมกว่า 90% (ผล Sprint 1) ในขณะที่เสียงกัดฟันเด่นชัดในย่านสูง (SNR 15.6 dB)
เกณฑ์ปรับตัวได้จาก `median + k×MAD` ของหน้าต่างเลื่อน 5 นาทีย้อนหลัง พร้อม
hysteresis กันสัญญาณกระพริบที่ขอบ threshold — เป็นไลบรารีล้วน (`src/`)
สตรีมทีละ block จริง ไม่โหลดทั้งคืนเข้า RAM ดูเหตุผลเชิงเทคนิคเต็มๆ
(ทำไมใช้ causal filter, ทำไม threshold คำนวณใหม่ทุก ~1 วินาทีไม่ใช่ทุกเฟรม,
บั๊ก hysteresis ที่ tests จับได้) ใน [`CLAUDE.md`](CLAUDE.md)

### 3. รัน detection — `tools/detect_events.py`

```bash
python -m tools.detect_events --session data/raw/sleep/ext_20260820
python -m tools.detect_events --session <path> --k 5 --min-duration 0.3
```

สร้างที่ `results/events/<session_id>/`:

- **`candidates.csv`** — event ที่ตรวจจับได้ทั้งหมด พร้อมเวลา, RMS,
  SNR เหนือ noise floor, สัดส่วนพลังงานครึ่งบนของย่าน ฯลฯ
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

ฐานของเครื่องมือติด label ที่ทันตแพทย์จะใช้ใน Sprint ถัดไป — เลือก session
(แสดง provenance กำกับ), เรียงคลิปตาม peak RMS มากไปน้อยเป็นค่าเริ่มต้น
(เจอของจริงเร็วสุด), มี spectrogram + ปุ่มขยายเสียง 1x/5x/20x (เสียงจริง
เบามาก), ปุ่ม label กัดฟัน/กรน/อื่นๆ/ไม่แน่ใจ บันทึกลง `labels.csv` ทันที
ทุกครั้งที่กด กลับมาทำต่อจากที่ค้างได้ อ่านเฉพาะจาก `results/events/` —
ไม่แตะไฟล์ session ต้นฉบับเลย

## โครงสร้างโค้ด

- `src/` — ไลบรารีล้วน (audio I/O, ฟีเจอร์เชิงสเปกตรัม, energy gate,
  schema ของ session.json) ไม่มี matplotlib/CLI/streamlit/argparse/side
  effect ใดๆ ออกแบบให้นำไปใช้ซ้ำกับ `tools/`, Streamlit, หรือแอป Android
  ในอนาคตได้โดยตรง
- `tools/` — CLI และโค้ดวาดกราฟ/Streamlit (`explore.py`,
  `import_external.py`, `detect_events.py`, `review.py`)
- `tests/` — pytest, ใช้สัญญาณสังเคราะห์เท่านั้น (`pytest` เพื่อรัน)
- `android/` — แอป Android สำหรับอัดเสียงจริง (ดูหัวข้อถัดไป)

## แอป Android

Sprint 1 พบว่าแอปอัดเสียงทั่วไปบันทึกเป็น m4a (AAC) แม้ตั้งนามสกุลเป็น .wav
และตัดความถี่เหนือ 8 kHz ทิ้งเกือบหมด — ทั้งที่ย่าน **8-16 kHz มี SNR 15.6
dB** ซึ่งเป็นย่านที่แยกเสียงถูฟันจากความเงียบได้ดีที่สุด (ดูเหตุผลเต็มใน
[`CLAUDE.md`](CLAUDE.md)) จึงต้องเขียนแอปอัดเสียงเองด้วย `AudioRecord` แบบ
PCM ดิบ ไม่ผ่านตัวเข้ารหัส/DSP ของผู้ผลิตเครื่อง

โครงสร้าง: Kotlin + Jetpack Compose, `minSdk 26`, `targetSdk 35`, package
`com.bruxecho.recorder`

### Build

ต้องมี JDK 17+ และ Android SDK (compileSdk 36, build-tools ที่รองรับ) —
ใช้ JBR ที่มากับ Android Studio ก็ได้ ไม่ต้องติดตั้ง JDK แยก:

```bash
cd android
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ./gradlew.bat :app:assembleDebug   # build อย่างเดียว
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ./gradlew.bat :app:installDebug     # build + ติดตั้งลงเครื่องที่ต่อ USB
```

`local.properties` (ระบุ path ของ Android SDK) เป็นไฟล์เฉพาะเครื่อง —
ไม่ถูก track โดย git ต้องสร้างเองถ้า clone ใหม่:

```
sdk.dir=C\:\\Users\\<user>\\AppData\\Local\\Android\\Sdk
```

### วิธีดึงไฟล์เสียงออกจากมือถือ

แอปเก็บไฟล์ไว้ที่ external app-specific storage (ไม่ต้องขอ permission บน
API 29+) — ดึงออกมาด้วย `adb pull` ได้ตรงๆ:

```bash
adb shell ls /storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions/
adb pull /storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions/<session_id>/ ./data/raw/
```

แต่ละคืนจะได้โฟลเดอร์ `sessions/<yyyyMMdd_HHmmss>/` ที่มี `segment_NNN.wav`
(แบ่งไฟล์ทุก ~1 ชั่วโมง) กับ `session.json` (เมทาดาทาสำหรับรายงาน — audio
source ที่ใช้จริง, sample rate, ช่วงที่ขาดหาย ฯลฯ)

### ตั้งค่า Samsung ก่อนใช้จริง (สำคัญ)

Samsung ปิดแอปพื้นหลังดุกว่าเครื่องอื่นมาก ถ้าไม่ตั้งค่าเพิ่ม แอปอาจถูกฆ่า
กลางดึกแล้วเสียข้อมูลทั้งคืน:

1. เปิดแอป BruxEcho → กดไอคอนตั้งค่า (มุมขวาบน) → กด **"ขอยกเว้นการประหยัด
   พลังงาน"**
2. ไปที่ Settings → Battery → Background usage limits → เพิ่ม **BruxEcho**
   ลงใน **"Never sleeping apps"**
3. ในหน้าเดียวกัน ปิด **"Put unused apps to sleep"** สำหรับแอปนี้

ก่อนนอนคืนแรก แนะนำให้เข้าหน้า **ตรวจสอบไมโครโฟน** (ไอคอนตั้งค่า) แล้วกด
"เริ่มทดสอบ 5 วินาที" หนึ่งครั้ง เพื่อยืนยันว่า audio source ที่ใช้จริงคือ
`UNPROCESSED` และย่าน 8-16 kHz ยังมีพลังงานเหลืออยู่ (ไม่ถูก DSP กรองทิ้ง)
ก่อนอัดจริง

### หมายเหตุทางเทคนิค

- ใช้ `AudioRecord` ไม่ใช่ `MediaRecorder` — ป้องกันการบีบอัดเป็น AAC ที่
  เป็นต้นตอของปัญหาใน Sprint 1
- audio source ไล่ลำดับ `UNPROCESSED` → `CAMCORDER` → `MIC` ตัวที่สร้าง
  `AudioRecord` สำเร็จ (state `STATE_INITIALIZED`) ตัวแรกจะถูกใช้ และบันทึก
  ไว้ใน `session.json` เป็น `audio_source_used`
- ปิด `AcousticEchoCanceler` / `NoiseSuppressor` / `AutomaticGainControl`
  ทุกตัวที่เครื่องรองรับ ก่อนเริ่มอัด
- segment ใหม่ทุก 1 ชั่วโมง, header ของไฟล์ WAV ที่กำลังเขียนถูกอัปเดตทุก
  ~30 วินาที เพื่อให้ไฟล์เปิดฟังได้แม้แอปถูกฆ่ากลางคัน

### โลโก้และไอคอน

โลโก้จริงมาจาก **`brand/logo.png`** (1500×1500px, ต้นฉบับ — ห้ามลบ/ห้ามแก้
ไฟล์นี้โดยตรง) เป็น lockup รวมฟันโครงร่าง + มอนแกรม "BEX" + คลื่นเสียง 4
ชั้น + wordmark "BruxEcho" อยู่ในภาพเดียวบนพื้นกรมท่าทึบ `#10162B`

`tools/regenerate_brand_assets.py` (`python -m
tools.regenerate_brand_assets`) crop/ประกอบภาพนี้เป็น asset จริง 3 ชุด
โดยอัตโนมัติ — ไม่ต้องแก้โค้ด Kotlin ใดๆ:

| Asset | ที่มา | ไปอยู่ที่ | หมายเหตุ |
|---|---|---|---|
| Launcher icon (adaptive foreground) | ฟัน + คลื่นเสียง (ตัด "BEX" ออก อ่านไม่ออกที่ขนาดเล็ก) | `res/mipmap-{m,h,xh,xxh,xxxh}dpi/ic_launcher_foreground.png` (108/162/216/324/432px) | จัดตำแหน่ง+ย่อขนาดอัตโนมัติให้พิกเซลทึบทุกจุดอยู่ในวงกลม safe-zone 66% เสมอ (ตรวจด้วยโค้ด ไม่ใช่ตาเปล่า) |
| Notification icon | ฟัน + คลื่นเสียงวงในสุดวงเดียว (ตัดวงนอกอีก 3 วงออก ลดรายละเอียด) | `res/drawable-{m,h,xh,xxh,xxxh}dpi/ic_notification.png` (24/36/48/72/96px) | บังคับ RGB ขาวล้วน เหลือแค่ alpha (silhouette) แล้ว dilate เส้นให้หนาขึ้นก่อนย่อ ไม่งั้น outline บางจะจางหายที่ 24px |
| Wordmark | ข้อความ "BruxEcho" ส่วนล่างของภาพ | `res/drawable/logo_wordmark.png` | ใช้ตรงจากต้นฉบับ ไม่ปรับสี |

`ic_launcher_background.xml` เป็น flat color (โทนเข้มจาก design system)
ไม่ใช่ raster เลยไม่ต้อง regenerate

**ถ้าจะเปลี่ยนโลโก้ในอนาคต:**
1. แทนที่ `brand/logo.png` ด้วยไฟล์ใหม่
2. เปิด `tools/regenerate_brand_assets.py` — ถ้าไฟล์ใหม่จัดวางองค์ประกอบ
   ต่างจากเดิม (ตำแหน่งฟัน/คลื่นเสียง/wordmark ในภาพ) ต้องหาพิกัด crop
   ใหม่ก่อน (ค่าคงที่ `TOOTH_BOX` / `WAVES_BOX` / `WORDMARK_BOX` ที่หัวไฟล์
   — วิธีหา: เปิดไฟล์ดูแล้วประเมิน bounding box คร่าวๆ ของแต่ละองค์ประกอบ)
3. รัน `python -m tools.regenerate_brand_assets`
4. `cd android && ./gradlew.bat :app:installDebug` แล้วตรวจตามขั้นตอนที่ 5
   ของ session ที่ทำครั้งแรก (launcher icon บนหน้า home จริง, notification
   icon ตอนอัดเสียง, wordmark บนหน้าหลัก)

ธีมสีปัจจุบัน (โทนอำพัน `#F0923C`) นิยามอยู่ที่
`ui/theme/Color.kt` — เป็นคนละสีกับโลโก้ (มิ้นท์ `#51D4BD`) โดยตั้งใจ ตาม
ที่ตัดสินใจไว้ตอนใส่โลโก้จริง: คงสีมิ้นท์ของแบรนด์ไว้ที่ตัว icon/wordmark
แต่ UI ในแอป (ปุ่ม, stat, ฯลฯ) ยังใช้อำพันตาม design system เดิม

### Design system (`ui/theme/`)

- **ธีมมืดเป็นค่าเริ่มต้นเสมอ** ไม่มีธีมสว่าง (`Theme.kt`)
- **สีหลักเป็นโทนอำพัน/ส้ม ไม่ใช่ฟ้า** — แสงสีฟ้าก่อนนอนรบกวนการหลั่ง
  เมลาโทนินมากกว่าแสงโทนอุ่นที่ความสว่างเท่ากัน (comment อธิบายไว้ใน
  `Theme.kt`)
- **Typography** (`Type.kt`) กำหนด line-height 1.6 เท่าของขนาดตัวอักษรทุก
  style (ปกติ Material ใช้ ~1.2x ซึ่งตัดสระบน/ล่างของภาษาไทย)
- **Spacing scale** (`Spacing.kt`): `xs=4 sm=8 md=12 lg=16 xl=24 xxl=32`
  (dp) — ใช้แทนตัวเลข padding ลอยๆ ในโค้ดใหม่ทุกจุด
- **ตัวเลขสำคัญ** (นาฬิกาตอนอัด, สถิติหน้าผลตรวจจับ) ใช้
  `tabularNumberStyle()` — monospace + OpenType `tnum` เพื่อไม่ให้ตัวเลข
  ขยับตำแหน่งตอนเปลี่ยนค่า

### หน้าผลการตรวจจับ / แนวโน้ม (Sprint 4 — ใช้ข้อมูลสมมติ)

**ยังไม่มีโมเดลตรวจจับจริง** หน้า "ผลการตรวจจับ" (`NightResultScreen`) และ
"แนวโน้ม" (`TrendsScreen`) จึงผูกกับ `DetectionRepository` interface
(`domain/DetectionRepository.kt`) แทนที่จะอ่านไฟล์ผลลัพธ์ตรงๆ — ตอนนี้มี
แค่ `MockDetectionRepository` ที่สร้างข้อมูลสมจริงจาก session จริงบนเครื่อง
(deterministic ตาม session id ไม่สุ่มใหม่ทุกครั้ง) Sprint 6 จะเพิ่ม
`RealDetectionRepository` ที่อ่าน events.csv จริงจาก `tools/detect_events.py`
โดย **UI ทั้งสองหน้าไม่ต้องแก้เลย** เพราะรู้จักแค่ interface

**โหมดสาธิต** (สวิตช์ในหน้าตั้งค่า) สลับไปใช้ชุดข้อมูลตัวอย่าง 14 คืนที่คุม
pattern ไว้ล่วงหน้า พร้อมคลิปเสียงตัวอย่างจริงใน
`app/src/main/assets/demo_clips/` (กดฟังได้จริงในหน้าผลการตรวจจับ) —
ใช้ตอนนำเสนอโดยไม่ต้องพึ่งว่าคืนก่อนหน้าบันทึกสำเร็จหรือไม่ มีป้าย "โหมด
สาธิต" กำกับไว้ตลอดเวลาที่เปิดใช้งาน กันสับสนกับข้อมูลจริง
