"""
dreamsea_collect.py
Searches the Dreamsea (HMML Cloud) catalog for manuscripts matching arbitrary
search-form fields, extracts the IIIF manifest URL from each manuscript's
detail page, and writes the URLs to a file for use with
scripts/collect_manuscripts.py.

Dreamsea's catalog is served at https://www.hmmlcloud.org/dreamsea/manuscripts.php
with a simple GET-style search form. Each manuscript's detail page embeds a
Mirador viewer whose manifestUri points to a IIIF v2 manifest on vhmml.org.
These manifests are publicly accessible and fully IIIF-compliant, so
collect_manuscripts.py can download the page images directly.

Search form fields supported (all optional, combined with AND):
    --language        e.g. "javanese" (loose filter — includes secondary languages)
    --script          e.g. "Javanese" (Aksara Jawa specifically — stricter)
    --country         e.g. "Indonesia"
    --city            e.g. "Indramayu (Jawa Barat)"
    --author
    --library
    --tags            subject matter (free text)
    --project_number  Dreamsea project ID
    --title
    --writing_support e.g. "palm leaf", "European paper"

Usage:
    # Recommended: filter by script=Javanese for actual Aksara Jawa content
    uv run python scripts/dreamsea_collect.py \\
        --script Javanese \\
        --country Indonesia \\
        --limit 50 \\
        --output annotation/dreamsea_manifests.txt

    # Narrow to a single collection (Indramayu, West Java)
    uv run python scripts/dreamsea_collect.py \\
        --script Javanese \\
        --city "Indramayu (Jawa Barat)" \\
        --output annotation/dreamsea_manifests.txt

    # Then download images:
    uv run python scripts/collect_manuscripts.py \\
        --manifests annotation/dreamsea_manifests.txt \\
        --output data/real/ \\
        --max_pages_per_manifest 5
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


SEARCH_URL = "https://www.hmmlcloud.org/dreamsea/manuscripts.php"
DETAIL_URL = "https://www.hmmlcloud.org/dreamsea/detail.php"
USER_AGENT = (
    "paddleocr-aksara-jawa/0.1 (research; https://github.com/setilanaji/paddleocr-aksara-jawa)"
)
REQUEST_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.3

MSID_RE = re.compile(r'detail\.php\?msid=([^"\' &]+)')
# Extract the IIIF manifest URL from the metadata table
#   <dt>IIIF Manifest:</dt><dd class='col-sm-9'>https://www.vhmml.org/image/manifest/834310</dd>
MANIFEST_RE = re.compile(
    r'IIIF Manifest[^<]*</dt>\s*<dd[^>]*>\s*(https?://[^\s<]+)',
    re.IGNORECASE,
)


def http_get_text(url: str, params: dict | None = None) -> str:
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.text


def search_msids(params: dict, limit: int) -> list[str]:
    """Return the list of msids matching the search form params."""
    # Drop empty values — the form accepts them but cleaner URLs for debugging.
    clean = {k: v for k, v in params.items() if v}
    clean.setdefault("searchType", "1")
    html = http_get_text(SEARCH_URL, params=clean)
    ids = list(dict.fromkeys(MSID_RE.findall(html)))  # dedupe, preserve order
    if limit:
        ids = ids[:limit]
    return ids


def manifest_for_msid(msid: str) -> str | None:
    """Fetch the manuscript detail page and return its IIIF manifest URL."""
    html = http_get_text(DETAIL_URL, params={"msid": msid})
    m = MANIFEST_RE.search(html)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(
        description="Extract Dreamsea IIIF manifest URLs via search-form filters.",
    )
    # Dreamsea search-form fields (all optional)
    ap.add_argument("--language",        default=None, help="Dreamsea language filter (loose — includes secondary)")
    ap.add_argument("--script",          default=None, help="Script filter (stricter — e.g. 'Javanese' for Aksara Jawa)")
    ap.add_argument("--country",         default=None)
    ap.add_argument("--city",            default=None, help='e.g. "Indramayu (Jawa Barat)"')
    ap.add_argument("--author",          default=None)
    ap.add_argument("--library",         default=None)
    ap.add_argument("--tags",            default=None, help="Subject-matter free text")
    ap.add_argument("--project_number",  default=None)
    ap.add_argument("--title",           default=None)
    ap.add_argument("--writing_support", default=None, help='e.g. "palm leaf"')
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max manuscripts to include (0 = all matches, default: 0)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("annotation/dreamsea_manifests.txt"),
        help="Output file — one manifest URL per line (comments prefixed with # preserved)",
    )
    args = ap.parse_args()

    search_params = {
        "language":        args.language,
        "script":          args.script,
        "country":         args.country,
        "city":            args.city,
        "author":          args.author,
        "library":         args.library,
        "tags":            args.tags,
        "projnum":         args.project_number,
        "title":           args.title,
        "writingSupport":  args.writing_support,
    }
    active = {k: v for k, v in search_params.items() if v}
    if not active:
        print("ERROR: provide at least one search filter (e.g. --script Javanese)", file=sys.stderr)
        sys.exit(1)

    print(f"Searching Dreamsea: {active}  limit={args.limit or 'all'}")
    try:
        msids = search_msids(search_params, args.limit)
    except requests.RequestException as e:
        print(f"ERROR: search failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(msids)} manuscripts — resolving manifest URLs...")

    filter_summary = " ".join(f"{k}={v!r}" for k, v in active.items())
    lines: list[str] = [
        f"# Dreamsea manifests — {filter_summary}  (scripts/dreamsea_collect.py)",
        "",
    ]
    resolved = 0
    for msid in tqdm(msids, unit="ms", desc="detail pages"):
        try:
            manifest = manifest_for_msid(msid)
        except requests.RequestException as e:
            print(f"  SKIP msid={msid}: {e}", file=sys.stderr)
            continue
        if not manifest:
            print(f"  SKIP msid={msid}: no manifestUri in detail page", file=sys.stderr)
            continue
        lines.append(f"# msid={msid}  detail={DETAIL_URL}?msid={msid}")
        lines.append(manifest)
        resolved += 1
        time.sleep(RATE_LIMIT_SLEEP)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Resolved {resolved} / {len(msids)} manifests")
    print(f"Output: {args.output.resolve()}")
    print()
    print("Next step — download images:")
    print("  uv run python scripts/collect_manuscripts.py \\")
    print(f"      --manifests {args.output} \\")
    print("      --output data/real/ --max_pages_per_manifest 5")


if __name__ == "__main__":
    main()
