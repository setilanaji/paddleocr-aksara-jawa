"""
build_triage_page.py
Generates annotation/triage.html — a static grid of thumbnails from data/real/
with keep/drop checkboxes for fast manual triage before annotation.

The HTML is fully client-side (no server, no build step) and uses localStorage
so closing and reopening the page resumes where you left off. A "Download
decisions" button saves a decisions.json that scripts/apply_triage.py consumes.

Usage:
    uv run python scripts/build_triage_page.py \\
        --image_dir data/real/ \\
        --output annotation/triage.html

Then open annotation/triage.html in a browser.
"""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_IMAGE_DIR = Path("data/real/")
DEFAULT_OUTPUT = Path("annotation/triage.html")


def load_sources_meta(image_dir: Path) -> dict[str, dict]:
    """Read sources.csv for per-image provenance to show in the UI."""
    sources_csv = image_dir / "sources.csv"
    if not sources_csv.exists():
        return {}
    meta: dict[str, dict] = {}
    with sources_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("local_filename", "")
            if fn:
                meta[fn] = row
    return meta


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Aksara Jawa — Image Triage ({total} images)</title>
<style>
  :root {{
    color-scheme: light dark;
    --keep-bg: #e8f5e9;
    --drop-bg: #ffebee;
    --keep-border: #4caf50;
    --drop-border: #f44336;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    padding: 16px 16px 120px 16px;
    background: #fafafa;
  }}
  header {{
    position: sticky;
    top: 0;
    background: rgba(250,250,250,0.95);
    backdrop-filter: blur(8px);
    padding: 12px 0;
    margin: -16px -16px 16px -16px;
    padding-left: 16px;
    padding-right: 16px;
    border-bottom: 1px solid #ddd;
    z-index: 10;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
  .stats {{ font-size: 14px; color: #555; }}
  .stats strong {{ color: #2e7d32; }}
  .stats .drop-count {{ color: #c62828; }}
  .controls {{ margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }}
  button {{
    padding: 6px 12px;
    font-size: 13px;
    border: 1px solid #aaa;
    background: white;
    border-radius: 4px;
    cursor: pointer;
  }}
  button:hover {{ background: #f0f0f0; }}
  button.primary {{ background: #1976d2; color: white; border-color: #1565c0; }}
  button.primary:hover {{ background: #1565c0; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: white;
    border: 2px solid transparent;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s;
  }}
  .card.keep {{ border-color: var(--keep-border); background: var(--keep-bg); }}
  .card.drop {{ border-color: var(--drop-border); background: var(--drop-bg); opacity: 0.55; }}
  .card img {{
    width: 100%;
    height: 260px;
    object-fit: contain;
    background: #fff;
    display: block;
    cursor: pointer;
  }}
  .card .meta {{
    padding: 6px 8px;
    font-size: 11px;
    color: #555;
    word-break: break-all;
    border-top: 1px solid #eee;
  }}
  .card .meta .fn {{ font-weight: 600; color: #222; }}
  .card label {{
    padding: 6px 8px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }}
  footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-top: 1px solid #ddd;
    padding: 10px 16px;
    display: flex;
    gap: 10px;
    align-items: center;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
  }}
  footer .hint {{ font-size: 12px; color: #666; margin-left: auto; }}
</style>
</head>
<body>
<header>
  <h1>Aksara Jawa — Image Triage</h1>
  <div class="stats">
    Total: <span id="total">{total}</span> &middot;
    Kept: <strong id="keep-count">0</strong> &middot;
    Dropped: <span class="drop-count" id="drop-count">0</span>
  </div>
  <div class="controls">
    <button onclick="selectAll(true)">Keep all</button>
    <button onclick="selectAll(false)">Drop all</button>
    <button onclick="invertSelection()">Invert</button>
  </div>
</header>

<div class="grid" id="grid"></div>

<footer>
  <button class="primary" onclick="downloadDecisions()">Download decisions.json</button>
  <button onclick="clearState()">Reset saved state</button>
  <span class="hint">Click the image (or checkbox) to toggle. Progress saves to localStorage automatically.</span>
</footer>

<script>
const IMAGES = {images_json};
const STORAGE_KEY = "aksara_triage_v1";

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
  }} catch (e) {{
    return {{}};
  }}
}}

function saveState(state) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

let state = loadState();
// Default new images to keep=true
for (const img of IMAGES) {{
  if (!(img.filename in state)) state[img.filename] = true;
}}

function render() {{
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const img of IMAGES) {{
    const keep = state[img.filename] !== false;
    const card = document.createElement("div");
    card.className = "card " + (keep ? "keep" : "drop");
    card.innerHTML = `
      <img src="${{img.src}}" loading="lazy" alt="${{img.filename}}">
      <label>
        <input type="checkbox" ${{keep ? "checked" : ""}} data-fn="${{img.filename}}">
        Keep
      </label>
      <div class="meta">
        <div class="fn">${{img.filename}}</div>
        <div>${{img.manifest_url || ""}}</div>
      </div>
    `;
    const checkbox = card.querySelector("input");
    const image = card.querySelector("img");
    const toggle = () => {{
      state[img.filename] = !state[img.filename];
      checkbox.checked = state[img.filename];
      saveState(state);
      card.className = "card " + (state[img.filename] ? "keep" : "drop");
      updateStats();
    }};
    checkbox.addEventListener("change", toggle);
    image.addEventListener("click", toggle);
    grid.appendChild(card);
  }}
  updateStats();
}}

function updateStats() {{
  let keep = 0, drop = 0;
  for (const img of IMAGES) (state[img.filename] !== false) ? keep++ : drop++;
  document.getElementById("keep-count").textContent = keep;
  document.getElementById("drop-count").textContent = drop;
}}

function selectAll(val) {{
  for (const img of IMAGES) state[img.filename] = val;
  saveState(state);
  render();
}}

function invertSelection() {{
  for (const img of IMAGES) state[img.filename] = !state[img.filename];
  saveState(state);
  render();
}}

function clearState() {{
  if (!confirm("Reset all triage decisions?")) return;
  localStorage.removeItem(STORAGE_KEY);
  state = {{}};
  for (const img of IMAGES) state[img.filename] = true;
  render();
}}

function downloadDecisions() {{
  const keep = IMAGES.filter(img => state[img.filename] !== false).map(img => img.filename);
  const drop = IMAGES.filter(img => state[img.filename] === false).map(img => img.filename);
  const payload = {{
    generated_at: new Date().toISOString(),
    total: IMAGES.length,
    keep, drop,
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "triage_decisions.json";
  a.click();
  URL.revokeObjectURL(url);
}}

render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Build a static HTML triage grid for data/real/ images.",
    )
    ap.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--glob",
        default="*.jpg",
        help="Image glob inside --image_dir (default: *.jpg)",
    )
    args = ap.parse_args()

    image_dir = args.image_dir.resolve()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob(args.glob))
    if not images:
        print(f"ERROR: no images found in {image_dir} matching {args.glob!r}")
        return

    meta = load_sources_meta(image_dir)

    # Build JS array with paths relative to the HTML file
    out_dir_resolved = output.parent.resolve()
    records = []
    for p in images:
        try:
            rel = p.resolve().relative_to(out_dir_resolved)
        except ValueError:
            # Fallback to a generic relative walk up
            rel = Path("..") / p.resolve().relative_to(out_dir_resolved.parent)
        m = meta.get(p.name, {})
        records.append({
            "filename":     p.name,
            "src":          str(rel),
            "manifest_url": m.get("manifest_url", ""),
            "canvas_label": m.get("canvas_label", ""),
        })

    html = HTML_TEMPLATE.format(
        total=len(records),
        images_json=json.dumps(records, ensure_ascii=False),
    )
    output.write_text(html, encoding="utf-8")

    print(f"Wrote triage page with {len(records)} images")
    print(f"  → {output.resolve()}")
    print(f"  Open with: open {output}  (macOS)  or  xdg-open {output}  (Linux)")
    print("\nWorkflow:")
    print("  1. Open the HTML file in your browser")
    print("  2. Click images to toggle keep/drop (progress auto-saves to localStorage)")
    print("  3. Download decisions.json")
    print("  4. uv run python scripts/apply_triage.py --decisions <path-to-json>")


if __name__ == "__main__":
    main()
