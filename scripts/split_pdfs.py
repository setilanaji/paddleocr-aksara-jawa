"""
split_pdfs.py
Extracts page images from a folder of local PDF files into data/real/ for
annotation.

Designed to pair with scripts/khastara_collect.py: that script produces a
shopping list of PDF URLs, you download them manually (Cloudflare blocks
programmatic access to Khastara), and this script turns them into page JPGs.

Naming:
  data/real/aksara_real_{NNNN}.jpg — continues from the highest existing index.

Provenance:
  data/real/sources.csv rows are appended per page with:
    local_filename, source_url, manifest_url, canvas_label
  source_url is file://<absolute PDF path>#page=N for idempotent reruns.

Re-running the script is idempotent — pages already in sources.csv are skipped.

Usage:
    uv run python scripts/split_pdfs.py \\
        --input ~/Downloads/khastara_pdfs/ \\
        --output data/real/

    # Extract only first 3 pages per PDF, lower DPI
    uv run python scripts/split_pdfs.py \\
        --input ~/Downloads/khastara_pdfs/ \\
        --output data/real/ \\
        --pages_per_pdf 3 \\
        --dpi 120
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import fitz
from PIL import Image
from tqdm import tqdm


SOURCES_CSV = "sources.csv"
FILENAME_PREFIX = "aksara_real"


def next_index(output_dir: Path) -> int:
    pattern = re.compile(rf"^{FILENAME_PREFIX}_(\d+)\.jpg$")
    max_idx = 0
    for p in output_dir.glob(f"{FILENAME_PREFIX}_*.jpg"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_seen_urls(sources_csv: Path) -> set[str]:
    if not sources_csv.exists():
        return set()
    seen: set[str] = set()
    with sources_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("source_url", "").strip()
            if url:
                seen.add(url)
    return seen


def append_source(sources_csv: Path, row: dict) -> None:
    exists = sources_csv.exists()
    with sources_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["local_filename", "source_url", "manifest_url", "canvas_label"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_jpg(img: Image.Image, dest: Path, width: int) -> None:
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    if width > 0 and img.width > width:
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
    img.save(dest, format="JPEG", quality=92, optimize=True)


def main():
    ap = argparse.ArgumentParser(
        description="Extract page images from a folder of PDFs into data/real/.",
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing PDF files (non-recursive)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/"),
        help="Output directory (default: data/real/)",
    )
    ap.add_argument(
        "--pages_per_pdf",
        type=int,
        default=0,
        help="Max pages per PDF (0 = all, default: 0)",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    ap.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Target image width in px (default: 1200, 0 = original)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input folder not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    sources_csv = args.output / SOURCES_CSV
    seen = load_seen_urls(sources_csv)
    idx = next_index(args.output)
    start_idx = idx
    width = args.width if args.width > 0 else 0

    pdfs = sorted(args.input.glob("*.pdf")) + sorted(args.input.glob("*.PDF"))
    if not pdfs:
        print(f"No PDFs found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Input folder   : {args.input.resolve()}  ({len(pdfs)} PDFs)")
    print(f"Output folder  : {args.output.resolve()}")
    print(f"Starting index : {idx:04d}")
    print(f"Pages per PDF  : {args.pages_per_pdf or 'all'}  @ {args.dpi} DPI")
    print(f"Target width   : {width or 'original'}px")
    print()

    for pdf_path in tqdm(pdfs, unit="pdf", desc="PDFs"):
        abs_path = pdf_path.resolve()
        try:
            doc = fitz.open(pdf_path)
        except (RuntimeError, ValueError) as e:
            print(f"  SKIP {pdf_path.name}: {e}", file=sys.stderr)
            continue

        try:
            n = min(len(doc), args.pages_per_pdf) if args.pages_per_pdf > 0 else len(doc)
            for page_num in range(n):
                source_url = f"file://{abs_path}#page={page_num + 1}"
                if source_url in seen:
                    continue

                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=args.dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                filename = f"{FILENAME_PREFIX}_{idx:04d}.jpg"
                save_jpg(img, args.output / filename, width)
                append_source(
                    sources_csv,
                    {
                        "local_filename": filename,
                        "source_url": source_url,
                        "manifest_url": str(abs_path),
                        "canvas_label": f"{pdf_path.stem}/page_{page_num + 1}",
                    },
                )
                seen.add(source_url)
                idx += 1
        finally:
            doc.close()

    total_rows = sum(1 for _ in sources_csv.open(encoding="utf-8")) - 1 if sources_csv.exists() else 0
    print()
    print(f"Extracted this run : {idx - start_idx}")
    print(f"Total sources.csv  : {total_rows}")


if __name__ == "__main__":
    main()
