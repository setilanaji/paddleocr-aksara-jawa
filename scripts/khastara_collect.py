"""
khastara_collect.py
Searches the PNRI Khastara catalog for Aksara Jawa manuscripts and writes a
"shopping list" CSV of browser URLs + file URLs.

Khastara's storage server (file-opac.perpusnas.go.id) is gated by Cloudflare
and rejects programmatic downloads. This script does NOT try to download —
it finds matching manuscripts via the public API and emits URLs for an
annotator to download manually in a browser (3 clicks per manuscript).

After you've downloaded the PDFs locally, use scripts/split_pdfs.py to
extract pages into data/real/.

API endpoints used (all public, no auth required):
  - /inlis/collection-list?subject=...   paginated search
  - /inlis/collection-files?id=...       per-manuscript file URLs

Output:
  A CSV with columns: catalog_id, title, language, konten_type, file_size,
  browser_url, file_url

Usage:
    # Default: Javanese manuscripts (subject "Manuskrip Jawa"), 50 rows, Indonesian default
    uv run python scripts/khastara_collect.py \\
        --output annotation/khastara_shopping_list.csv

    # Narrow to 20 manuscripts
    uv run python scripts/khastara_collect.py \\
        --manuscripts_limit 20 \\
        --output annotation/khastara_shopping_list.csv

    # Different subject
    uv run python scripts/khastara_collect.py \\
        --subject "Kesusastraan Jawa" \\
        --output annotation/shopping_sastra.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


API_BASE = "https://khastara-api.perpusnas.go.id"
CATALOG_WEB_BASE = "https://khastara.perpusnas.go.id/koleksi-digital"
USER_AGENT = (
    "paddleocr-aksara-jawa/0.1 (research; https://github.com/setilanaji/paddleocr-aksara-jawa)"
)
REQUEST_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.5


def http_get_json(url: str) -> dict:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def search_manuscripts(subject: str, limit: int) -> list[dict]:
    """Walk paginated collection-list until we have `limit` items (or exhaust)."""
    items: list[dict] = []
    per_page = min(100, limit) if limit else 100
    page = 1
    while True:
        url = (
            f"{API_BASE}/inlis/collection-list"
            f"?subject={requests.utils.quote(subject)}"
            f"&per_page={per_page}&page={page}"
        )
        data = http_get_json(url)
        page_items = data.get("data", [])
        if not page_items:
            break
        items.extend(page_items)
        total = data.get("meta", {}).get("total", 0)
        if limit and len(items) >= limit:
            items = items[:limit]
            break
        if len(items) >= total:
            break
        page += 1
        time.sleep(RATE_LIMIT_SLEEP)
    return items


def fetch_files(catalog_id: int) -> list[dict]:
    url = f"{API_BASE}/inlis/collection-files?id={catalog_id}"
    data = http_get_json(url)
    return data.get("data", {}).get("files", [])


def main():
    ap = argparse.ArgumentParser(
        description="Emit a shopping list of Khastara manuscript URLs for manual download.",
    )
    ap.add_argument("--subject", default="Manuskrip Jawa", help="Subject facet (default: 'Manuskrip Jawa')")
    ap.add_argument(
        "--manuscripts_limit",
        type=int,
        default=50,
        help="Max manuscripts to list (0 = no cap, default: 50)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("annotation/khastara_shopping_list.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    print(f"Searching Khastara: subject={args.subject!r}  limit={args.manuscripts_limit or 'none'}")
    try:
        items = search_manuscripts(args.subject, args.manuscripts_limit)
    except requests.RequestException as e:
        print(f"ERROR: search failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(items)} manuscripts — resolving file URLs...")

    rows: list[dict] = []
    for item in tqdm(items, unit="ms", desc="resolving"):
        catalog_id = item.get("catalog_id")
        try:
            files = fetch_files(int(catalog_id))
        except requests.RequestException as e:
            print(f"  SKIP {catalog_id}: {e}", file=sys.stderr)
            continue
        time.sleep(RATE_LIMIT_SLEEP)

        for fg in files:
            konten_type = fg.get("konten_type", "")
            entries = fg.get("digital_konten_list", [])
            for entry in entries:
                rows.append(
                    {
                        "catalog_id": catalog_id,
                        "title": (item.get("title") or "").strip(),
                        "language_code": item.get("language_code") or "",
                        "konten_type": konten_type,
                        "file_size": fg.get("all_file_size", ""),
                        "browser_url": f"{CATALOG_WEB_BASE}/{catalog_id}",
                        "file_url": entry.get("file_konten_url", "").strip(),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "catalog_id",
                "title",
                "language_code",
                "konten_type",
                "file_size",
                "browser_url",
                "file_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summary by file type
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["konten_type"]] = by_type.get(r["konten_type"], 0) + 1

    print()
    print(f"Wrote {len(rows)} file entries to {args.output.resolve()}")
    print("Breakdown by konten_type:")
    for kt, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {kt or '(empty)'}: {n}")
    print()
    print("Next steps:")
    print(f"  1. Open {args.output} and review")
    print("  2. For each manuscript, open browser_url in your browser and click Download")
    print("     (Cloudflare blocks programmatic downloads)")
    print("  3. Save downloaded PDFs to a folder, e.g. ~/Downloads/khastara_pdfs/")
    print("  4. Extract pages with:")
    print("     uv run python scripts/split_pdfs.py \\")
    print("         --input ~/Downloads/khastara_pdfs/ --output data/real/")


if __name__ == "__main__":
    main()
