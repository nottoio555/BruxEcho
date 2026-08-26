#!/usr/bin/env bash
# Records 30s through the BruxEcho app on a connected Android device, pulls
# the result, and checks it end-to-end: correct WAV format (ffprobe) and
# real energy surviving in the 8+ kHz band (tools/explore.py) — the two
# things a generic recorder app got wrong in Sprint 1 (see CLAUDE.md).
#
# Usage: scripts/verify_recording.sh [device-serial]
set -euo pipefail
# Git Bash on Windows auto-converts POSIX-looking absolute paths in argv to
# Windows paths before calling native .exe tools like adb — which mangles
# on-device paths like /storage/emulated/... Exclude just that prefix so
# local temp-dir paths (which DO need the normal conversion) still work.
export MSYS2_ARG_CONV_EXCL="/storage"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="com.bruxecho.recorder"
RECORD_SECONDS=30
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

SERIAL="${1:-}"
ADB=(adb)
if [[ -n "$SERIAL" ]]; then
  ADB=(adb -s "$SERIAL")
fi

pass() { echo "✓ $1"; }
fail() { echo "✗ $1"; OVERALL_PASS=0; }
OVERALL_PASS=1

echo "== ตรวจสอบอุปกรณ์ =="
if ! "${ADB[@]}" get-state >/dev/null 2>&1; then
  echo "ไม่พบอุปกรณ์ที่เชื่อมต่อ (adb devices ว่างเปล่า หรือระบุ serial ผิด)"
  exit 1
fi
pass "พบอุปกรณ์"

echo "== เปิดแอปและสั่งอัด ${RECORD_SECONDS}s =="
read -r WIDTH HEIGHT < <("${ADB[@]}" shell wm size | sed -E 's/.*: ([0-9]+)x([0-9]+)/\1 \2/')
TAP_X=$(( WIDTH / 2 ))
START_TAP_Y=$(( HEIGHT * 52 / 100 ))
STOP_TAP_Y=$(( HEIGHT * 68 / 100 ))

"${ADB[@]}" shell am force-stop "$PACKAGE" >/dev/null # guarantee a clean idle-screen start regardless of prior state
"${ADB[@]}" shell am start -n "$PACKAGE/.ui.MainActivity" >/dev/null
sleep 2
"${ADB[@]}" shell input tap "$TAP_X" "$START_TAP_Y"
echo "  (ถ้าเป็นการอัดครั้งแรก จะมี dialog ขอสิทธิ์ไมโครโฟนโผล่ขึ้นมา —"
echo "   สคริปต์นี้ไม่ auto-grant ให้ กรุณากดอนุญาตเองภายใน 5 วินาที)"
sleep 5
"${ADB[@]}" shell input tap "$TAP_X" "$START_TAP_Y" >/dev/null 2>&1 || true # retry once in case the permission dialog ate the first tap
echo "  กำลังอัด..."
sleep "$RECORD_SECONDS"
"${ADB[@]}" shell input tap "$TAP_X" "$STOP_TAP_Y"
sleep 2

echo "== ดึงไฟล์ล่าสุดออกจากเครื่อง =="
DEVICE_SESSIONS="/storage/emulated/0/Android/data/$PACKAGE/files/sessions"
LATEST_SESSION="$("${ADB[@]}" shell "ls -1 $DEVICE_SESSIONS 2>/dev/null | sort | tail -1" | tr -d '\r')"
if [[ -z "$LATEST_SESSION" ]]; then
  fail "ไม่พบโฟลเดอร์ session บนเครื่อง — การอัดอาจไม่สำเร็จ (ตรวจสอบว่าอนุญาตไมโครโฟนแล้ว)"
  exit 1
fi
pass "session ล่าสุด: $LATEST_SESSION"

PULL_DIR="$WORK_DIR/$LATEST_SESSION"
mkdir -p "$PULL_DIR"
"${ADB[@]}" pull "$DEVICE_SESSIONS/$LATEST_SESSION/" "$WORK_DIR/" >/dev/null

WAV_FILE="$(find "$PULL_DIR" -name 'segment_*.wav' | sort | head -1)"
if [[ -z "$WAV_FILE" ]]; then
  fail "ไม่พบไฟล์ segment_*.wav ใน session ที่ดึงมา"
  exit 1
fi
pass "ดึงไฟล์แล้ว: $WAV_FILE"

echo "== ตรวจสอบฟอร์แมตด้วย ffprobe =="
if ! command -v ffprobe >/dev/null 2>&1; then
  fail "ไม่พบ ffprobe ใน PATH — ข้ามการตรวจสอบฟอร์แมต"
else
  PROBE="$(ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 "$WAV_FILE")"
  echo "$PROBE"
  CODEC="$(echo "$PROBE" | sed -n 's/codec_name=//p')"
  RATE="$(echo "$PROBE" | sed -n 's/sample_rate=//p')"
  CHANNELS="$(echo "$PROBE" | sed -n 's/channels=//p')"

  [[ "$CODEC" == "pcm_s16le" ]] && pass "codec = pcm_s16le" || fail "codec = $CODEC (ต้องการ pcm_s16le)"
  [[ "$RATE" == "44100" ]] && pass "sample rate = 44100 Hz" || fail "sample rate = $RATE Hz (ต้องการ 44100)"
  [[ "$CHANNELS" == "1" ]] && pass "mono (1 channel)" || fail "channels = $CHANNELS (ต้องการ 1)"
fi

echo "== ตรวจสอบพลังงานย่านความถี่สูงด้วย tools.explore =="
EXPLORE_OUT="$WORK_DIR/explore_out"
PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

( cd "$REPO_ROOT" && "$PYTHON" -m tools.explore --input "$PULL_DIR" --output "$EXPLORE_OUT" --trim-start 0 ) || true

SUMMARY_CSV="$EXPLORE_OUT/summary.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
  VERDICT_LINE="$("$PYTHON" - "$SUMMARY_CSV" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("0.0 LOW")
else:
    pct = float(rows[0]["high_freq_ratio_pct"])
    print(f"{pct} {'OK' if pct > 0.01 else 'LOW'}")
PY
)"
  HIGH_FREQ_PCT="${VERDICT_LINE%% *}"
  VERDICT="${VERDICT_LINE##* }"
  if [[ "$VERDICT" == "OK" ]]; then
    pass "พลังงานเหนือ 8 kHz = ${HIGH_FREQ_PCT}% ของพลังงานรวม (ไม่ถูกตัดทิ้ง)"
  else
    fail "พลังงานเหนือ 8 kHz = ${HIGH_FREQ_PCT}% — ต่ำผิดปกติ อาจถูก DSP กรองทิ้ง (หรือคลิปนี้เงียบจริงๆ ลองอีกครั้งพร้อมทำเสียงใกล้ไมค์)"
  fi
  echo "  รายละเอียดเต็ม: $SUMMARY_CSV"
else
  fail "tools.explore ไม่สร้าง summary.csv — ดู error ด้านบน"
fi

echo
if [[ "$OVERALL_PASS" -eq 1 ]]; then
  echo "=== ผ่านทั้งหมด ==="
else
  echo "=== มีรายการไม่ผ่าน ดูรายละเอียดด้านบน ==="
  exit 1
fi
