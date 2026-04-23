"""
apply_triage.py
Applies a decisions.json emitted by annotation/triage.html to data/real/:
  - deletes any image in the "drop" list
  - removes its row from sources.csv

Re-running with the same decisions.json is safe — missing images are skipped.

Usage:
    uv run python scripts/apply_triage.py \\
        --decisions ~/Downloads/triage_decisions.json \\
        --image_dir data/real/

Add --dry_run to preview without deleting.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Apply triage decisions.json — drop marked images + clean sources.csv.",
    )
    ap.add_argument("--decisions", type=Path, required=True, help="Path to decisions.json")
    ap.add_argument("--image_dir", type=Path, default=Path("data/real/"))
    ap.add_argument("--dry_run", action="store_true", help="Preview changes without writing")
    args = ap.parse_args()

    if not args.decisions.exists():
        print(f"ERROR: decisions file not found: {args.decisions}", file=sys.stderr)
        sys.exit(1)
    if not args.image_dir.exists():
        print(f"ERROR: image_dir not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    drop_list: list[str] = payload.get("drop", [])
    keep_list: list[str] = payload.get("keep", [])

    print(f"Decisions file : {args.decisions}")
    print(f"  Keep          : {len(keep_list)}")
    print(f"  Drop          : {len(drop_list)}")
    print(f"  Generated at  : {payload.get('generated_at', '?')}")
    print(f"  Image dir     : {args.image_dir.resolve()}")
    print(f"  Mode          : {'DRY RUN' if args.dry_run else 'APPLY'}\n")

    # Delete images
    deleted, missing = 0, 0
    for fn in drop_list:
        p = args.image_dir / fn
        if not p.exists():
            missing += 1
            continue
        if args.dry_run:
            print(f"  would delete  {p}")
        else:
            p.unlink()
        deleted += 1

    # Rewrite sources.csv
    sources_csv = args.image_dir / "sources.csv"
    csv_removed = 0
    if sources_csv.exists():
        drop_set = set(drop_list)
        with sources_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        kept_rows = [r for r in rows if r.get("local_filename") not in drop_set]
        csv_removed = len(rows) - len(kept_rows)

        if not args.dry_run and csv_removed > 0:
            with sources_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)

    print()
    print(f"Images {'would be ' if args.dry_run else ''}deleted : {deleted}")
    print(f"Images already missing            : {missing}")
    print(f"sources.csv rows {'would be ' if args.dry_run else ''}removed : {csv_removed}")


if __name__ == "__main__":
    main()
