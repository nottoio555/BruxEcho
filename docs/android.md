# รายละเอียดแอป Android

การ build ติดตั้ง และตั้งค่าก่อนใช้อยู่ใน [README](../README.md#แอป-android)
หน้านี้เก็บรายละเอียดเชิงเทคนิคและการออกแบบ

## หมายเหตุทางเทคนิค

- ใช้ `AudioRecord` ไม่ใช่ `MediaRecorder` — ป้องกันการบีบอัดเป็น AAC ที่
  เป็นต้นตอของปัญหาใน Sprint 1
- audio source ไล่ลำดับ `UNPROCESSED` → `CAMCORDER` → `MIC` ตัวที่สร้าง
  `AudioRecord` สำเร็จ (state `STATE_INITIALIZED`) ตัวแรกจะถูกใช้ และบันทึก
  ไว้ใน `session.json` เป็น `audio_source_used`
- ปิด `AcousticEchoCanceler` / `NoiseSuppressor` / `AutomaticGainControl`
  ทุกตัวที่เครื่องรองรับ ก่อนเริ่มอัด
- segment ใหม่ทุก 1 ชั่วโมง, header ของไฟล์ WAV ที่กำลังเขียนถูกอัปเดตทุก
  ~30 วินาที เพื่อให้ไฟล์เปิดฟังได้แม้แอปถูกฆ่ากลางคัน

## โหมดสาธิต

**โหมดสาธิต** (สวิตช์ในหน้าตั้งค่า) สลับไปใช้ชุดข้อมูลตัวอย่าง 14 คืนที่คุม
pattern ไว้ล่วงหน้า พร้อมคลิปเสียงตัวอย่างจริงใน
`app/src/main/assets/demo_clips/` (กดฟังได้จริงในหน้าผลการตรวจจับ) —
ใช้ตอนนำเสนอโดยไม่ต้องพึ่งว่าคืนก่อนหน้าบันทึกสำเร็จหรือไม่ มีป้าย "โหมด

## โลโก้และไอคอน

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

## Design system (`ui/theme/`)

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
