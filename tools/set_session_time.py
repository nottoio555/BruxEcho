"""แก้เวลาเริ่มของ session ที่ import ไปแล้ว โดยไม่ต้อง import ใหม่

ไฟล์เสียงไม่ถูกแตะเลย — แก้แค่ ``session.json`` ซึ่งเป็นตัวบอกว่า session นี้
ไปอยู่วันไหนบนปฏิทิน การ import ใหม่ทั้งชุดเพื่อเปลี่ยนเวลาอย่างเดียวต้อง
คัดลอกไฟล์เสียงหลายร้อย MB ใหม่หมด ทั้งที่ข้อมูลเสียงเหมือนเดิมทุกอย่าง

``ended_at`` จะถูกเลื่อนตามด้วยระยะเท่ากัน เพื่อให้ความยาวคืนคงเดิม — เว้นแต่
จะระบุ ``--ended-at`` มาเอง

วิธีใช้:
    python -m tools.set_session_time --session data/raw/sleep/dc_demo --started-at "2026-09-17 22:30"
    python -m tools.set_session_time --session data/raw/sleep/dc_demo --started-at "2026-09-17T22:30:00+07:00" --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("bruxecho.set_session_time")

ACCEPTED_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
)


def parse_timestamp(text: str, reference: datetime | None) -> datetime:
    """Parse a user-typed time, borrowing the session's timezone if none given.

    Borrowing rather than assuming UTC or local: the session already knows
    which offset its clock times were recorded in, and silently changing that
    would move the night to a different calendar day — the exact bug this
    tool exists to fix.
    """
    for fmt in ACCEPTED_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None and reference is not None and reference.tzinfo is not None:
            parsed = parsed.replace(tzinfo=reference.tzinfo)
        return parsed
    raise SystemExit(
        f"อ่านเวลา {text!r} ไม่ออก — ใช้รูปแบบเช่น '2026-09-17 22:30' "
        "หรือ '2026-09-17T22:30:00+07:00'"
    )


def read_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def run(session_dir: Path, started_at_text: str, ended_at_text: str | None, dry_run: bool) -> None:
    path = session_dir / "session.json"
    if not path.is_file():
        raise SystemExit(f"ไม่พบ {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    old_start = read_iso(data.get("started_at"))
    old_end = read_iso(data.get("ended_at"))
    if old_start is None:
        raise SystemExit("session.json ไม่มี started_at เดิม — แก้ไม่ได้")

    new_start = parse_timestamp(started_at_text, old_start)
    if ended_at_text:
        new_end = parse_timestamp(ended_at_text, old_start)
    elif old_end is not None:
        # Shift by the same delta so the night keeps its length.
        new_end = old_end + (new_start - old_start)
    else:
        new_end = None

    if new_end is not None and new_end <= new_start:
        raise SystemExit(
            f"ended_at ({new_end.isoformat()}) ต้องอยู่หลัง started_at ({new_start.isoformat()})"
        )

    print()
    print("=" * 70)
    print(f"session: {data.get('session_id')}  ({session_dir})")
    print("=" * 70)
    print(f"  started_at  {old_start.isoformat()}  ->  {new_start.isoformat()}")
    if old_end is not None or new_end is not None:
        old_text = old_end.isoformat() if old_end else "-"
        new_text = new_end.isoformat() if new_end else "-"
        print(f"  ended_at    {old_text}  ->  {new_text}")
    if old_end is not None and new_end is not None:
        kept = timedelta(seconds=(new_end - new_start).total_seconds())
        print(f"  ความยาวคืน  {kept} (คงเดิม)")
    print(f"  วันบนปฏิทิน {old_start.date()}  ->  {new_start.date()}")
    print()

    if dry_run:
        print("--dry-run — ยังไม่ได้เขียนอะไรลงไฟล์")
        print("=" * 70)
        return

    data["started_at"] = new_start.isoformat()
    if new_end is not None:
        data["ended_at"] = new_end.isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"เขียน {path} แล้ว (ไฟล์เสียงไม่ถูกแตะ)")
    print()
    print("ถ้า session นี้อยู่บนมือถือแล้ว ต้องส่ง session.json ใหม่ด้วย:")
    print(
        "  adb push "
        f"{path} /storage/emulated/0/Android/data/com.bruxecho.recorder/files/"
        f"sessions/{data.get('session_id')}/session.json"
    )
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="แก้ started_at ของ session ที่ import แล้ว")
    parser.add_argument("--session", required=True, type=Path, help="โฟลเดอร์ session")
    parser.add_argument("--started-at", required=True, help="เวลาเริ่มใหม่")
    parser.add_argument("--ended-at", default=None, help="เวลาจบใหม่ (ไม่ใส่ = เลื่อนตามให้ความยาวคงเดิม)")
    parser.add_argument("--dry-run", action="store_true", help="แสดงผลอย่างเดียว ไม่เขียนไฟล์")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.session, args.started_at, args.ended_at, args.dry_run)


if __name__ == "__main__":
    main()
