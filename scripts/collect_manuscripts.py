"""
collect_manuscripts.py
Downloads Aksara Jawa manuscript page images from IIIF manifests or direct
image URLs into data/real/ for annotation.

Supports:
  - IIIF Presentation API v2 and v3 manifests (Khastara, Dreamsea, EAP, most academic libraries)
  - Direct image URLs (JPG, PNG, TIFF)

Writes:
  - data/real/aksara_real_{NNNN}.jpg — images at requested width (default 1200px)
  - data/real/sources.csv — local_filename, source_url, manifest_url, canvas_label

Re-running the script skips already-downloaded files (keyed on source URL in sources.csv)
and appends new images starting from the next free index.

Usage:
    # From a file of IIIF manifest URLs (one per line)
    uv run python scripts/collect_manuscripts.py \\
        --manifests manifests.txt \\
        --output data/real/

    # From a file of direct image URLs
    uv run python scripts/collect_manuscripts.py \\
        --images urls.txt \\
        --output data/real/

    # Mixed + limit pages per manifest + custom width
    uv run python scripts/collect_manuscripts.py \\
        --manifests manifests.txt \\
        --images urls.txt \\
        --output data/real/ \\
        --max_pages_per_manifest 10 \\
        --width 1200
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm


USER_AGENT = (
    "paddleocr-aksara-jawa/0.1 (research; https://github.com/setilanaji/paddleocr-aksara-jawa)"
)
SOURCES_CSV = "sources.csv"
FILENAME_PREFIX = "aksara_real"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 2


class CollectError(Exception):
    pass


def http_get(url: str, stream: bool = False) -> requests.Response:
    """GET with retries on 5xx / network errors."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                stream=stream,
            )
            if r.status_code >= 500:
                raise CollectError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, CollectError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise CollectError(f"request failed for {url}: {last_exc}")


def iter_canvas_images(manifest: dict) -> Iterator[tuple[str, str | None, str]]:
    """
    Yield (image_url, iiif_service_base, canvas_label) for every canvas in a
    IIIF v2 or v3 manifest. `iiif_service_base` is the Image API base URL if
    present — preferred for resizing.
    """
    # v2: sequences[].canvases[].images[].resource
    sequences = manifest.get("sequences") or []
    for seq in sequences:
        for canvas in seq.get("canvases", []):
            label = canvas.get("label") or ""
            if isinstance(label, list):
                label = label[0] if label else ""
            for image in canvas.get("images", []):
                resource = image.get("resource") or {}
                image_url = resource.get("@id") or resource.get("id")
                service = resource.get("service") or {}
                if isinstance(service, list):
                    service = service[0] if service else {}
                service_base = service.get("@id") or service.get("id")
                if image_url:
                    yield image_url, service_base, str(label)

    # v3: items[] (Canvas) -> items[] (AnnotationPage) -> items[] (Annotation) -> body
    for canvas in manifest.get("items") or []:
        label = canvas.get("label") or ""
        if isinstance(label, dict):
            vals = next(iter(label.values()), [])
            label = vals[0] if vals else ""
        for ap in canvas.get("items", []):
            for anno in ap.get("items", []):
                body = anno.get("body") or {}
                if isinstance(body, list):
                    body = body[0] if body else {}
                image_url = body.get("id")
                service = body.get("service") or []
                if isinstance(service, dict):
                    service = [service]
                service_base = None
                if service:
                    service_base = service[0].get("@id") or service[0].get("id")
                if image_url:
                    yield image_url, service_base, str(label)


def iiif_resize_url(service_base: str, width: int) -> str:
    """Build a IIIF Image API URL at {width}px wide, JPG output."""
    base = service_base.rstrip("/")
    return f"{base}/full/{width},/0/default.jpg"


def next_index(output_dir: Path) -> int:
    """Next free NNNN given existing aksara_real_NNNN.jpg files."""
    pattern = re.compile(rf"^{FILENAME_PREFIX}_(\d+)\.jpg$")
    max_idx = 0
    for p in output_dir.glob(f"{FILENAME_PREFIX}_*.jpg"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_seen_urls(sources_csv: Path) -> set[str]:
    """Read source_url column from sources.csv (if exists) for deduplication."""
    if not sources_csv.exists():
        return set()
    seen: set[str] = set()
    with sources_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("source_url", "").strip()
            if url:
                seen.add(url)
    return seen


def append_source(sources_csv: Path, row: dict) -> None:
    """Append one row to sources.csv, creating header if needed."""
    exists = sources_csv.exists()
    with sources_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["local_filename", "source_url", "manifest_url", "canvas_label"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def download_to_jpg(url: str, dest: Path, width: int) -> None:
    """Download image bytes at URL, convert to JPG, resize to max width if width > 0."""
    r = http_get(url, stream=True)
    content = r.content

    with Image.open(io.BytesIO(content)) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        if width > 0 and img.width > width:
            ratio = width / img.width
            new_size = (width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        img.save(dest, format="JPEG", quality=92, optimize=True)


def read_url_list(path: Path) -> list[str]:
    """One URL per line; blank lines and '#' comments ignored."""
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def collect_from_manifests(
    manifest_urls: list[str],
    output_dir: Path,
    sources_csv: Path,
    seen: set[str],
    width: int,
    max_pages_per_manifest: int | None,
    start_idx: int,
) -> int:
    """Download images referenced by a list of IIIF manifest URLs. Returns new next index."""
    idx = start_idx
    for manifest_url in tqdm(manifest_urls, desc="manifests", unit="manifest"):
        try:
            r = http_get(manifest_url)
            manifest = r.json()
        except (CollectError, ValueError) as e:
            print(f"  SKIP manifest {manifest_url}: {e}", file=sys.stderr)
            continue

        pages = list(iter_canvas_images(manifest))
        if max_pages_per_manifest:
            pages = pages[:max_pages_per_manifest]

        for image_url, service_base, canvas_label in tqdm(
            pages, desc="  pages", unit="page", leave=False
        ):
            fetch_url = (
                iiif_resize_url(service_base, width) if (service_base and width > 0) else image_url
            )
            if fetch_url in seen or image_url in seen:
                continue

            filename = f"{FILENAME_PREFIX}_{idx:04d}.jpg"
            dest = output_dir / filename
            # If we used IIIF service to pre-size, no need to resize locally.
            local_resize = 0 if service_base and width > 0 else width
            try:
                download_to_jpg(fetch_url, dest, local_resize)
            except (CollectError, OSError, Image.UnidentifiedImageError) as e:
                print(f"  SKIP {fetch_url}: {e}", file=sys.stderr)
                continue

            append_source(
                sources_csv,
                {
                    "local_filename": filename,
                    "source_url": fetch_url,
                    "manifest_url": manifest_url,
                    "canvas_label": canvas_label,
                },
            )
            seen.add(fetch_url)
            idx += 1
    return idx


def collect_from_image_urls(
    image_urls: list[str],
    output_dir: Path,
    sources_csv: Path,
    seen: set[str],
    width: int,
    start_idx: int,
) -> int:
    """Download direct image URLs. Returns new next index."""
    idx = start_idx
    for url in tqdm(image_urls, desc="images", unit="image"):
        if url in seen:
            continue

        filename = f"{FILENAME_PREFIX}_{idx:04d}.jpg"
        dest = output_dir / filename
        try:
            download_to_jpg(url, dest, width)
        except (CollectError, OSError, Image.UnidentifiedImageError) as e:
            print(f"  SKIP {url}: {e}", file=sys.stderr)
            continue

        append_source(
            sources_csv,
            {
                "local_filename": filename,
                "source_url": url,
                "manifest_url": "",
                "canvas_label": urlparse(url).path.rsplit("/", 1)[-1],
            },
        )
        seen.add(url)
        idx += 1
    return idx


def main():
    ap = argparse.ArgumentParser(
        description="Download Aksara Jawa manuscript page images from IIIF manifests or direct URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--manifests",
        type=Path,
        help="Path to a text file of IIIF manifest URLs (one per line)",
    )
    ap.add_argument(
        "--images",
        type=Path,
        help="Path to a text file of direct image URLs (one per line)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/"),
        help="Output directory (default: data/real/)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Target image width in pixels (default: 1200, set 0 to keep original)",
    )
    ap.add_argument(
        "--max_pages_per_manifest",
        type=int,
        default=None,
        help="Cap pages downloaded per manifest (default: no cap)",
    )
    args = ap.parse_args()

    if not args.manifests and not args.images:
        print("ERROR: provide --manifests and/or --images", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_csv = output_dir / SOURCES_CSV

    seen = load_seen_urls(sources_csv)
    start_idx = next_index(output_dir)
    idx = start_idx
    width = args.width if args.width > 0 else 0

    print(f"Output directory  : {output_dir.resolve()}")
    print(f"Sources CSV       : {sources_csv.name}  ({len(seen)} URLs already seen)")
    print(f"Starting index    : {idx:04d}")
    print(f"Target width      : {width or 'original'}")
    print()

    if args.manifests:
        manifest_urls = read_url_list(args.manifests)
        print(f"Reading {len(manifest_urls)} manifest URLs from {args.manifests}")
        idx = collect_from_manifests(
            manifest_urls, output_dir, sources_csv, seen, width,
            args.max_pages_per_manifest, idx,
        )

    if args.images:
        image_urls = read_url_list(args.images)
        print(f"Reading {len(image_urls)} direct image URLs from {args.images}")
        idx = collect_from_image_urls(
            image_urls, output_dir, sources_csv, seen, width, idx,
        )

    new_count = idx - start_idx
    total_rows = sum(1 for _ in sources_csv.open(encoding="utf-8")) - 1 if sources_csv.exists() else 0
    print()
    print(f"Downloaded this run : {new_count}")
    print(f"Total sources.csv   : {total_rows}")


if __name__ == "__main__":
    main()
