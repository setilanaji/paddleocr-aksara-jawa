# Annotation Guide — Label Studio

Step-by-step setup for annotating real Aksara Jawa manuscript images at line level.
Produces output compatible with PaddleOCR `Label.txt` and ERNIEKit SFT format.

---

## 1. Prerequisites

- macOS, Linux, or Windows
- 4 GB free disk space
- One of:
  - **Docker Desktop** (recommended — zero Python setup)
  - **Python 3.11+** with `pip`

---

## 2. Start Label Studio

The stack (Label Studio + Cloudflare tunnel) is defined in
[`annotation/docker-compose.yml`](../annotation/docker-compose.yml).

```bash
# From the repository root
docker compose -f annotation/docker-compose.yml up -d

# Stop
docker compose -f annotation/docker-compose.yml down

# Logs
docker compose -f annotation/docker-compose.yml logs -f
```

This brings up two containers:

- `aksara-labelstudio` — Label Studio on `http://localhost:8080`
- `aksara-cloudflared` — Cloudflare tunnel publishing `https://label.yourdomain.com`

### One-time tunnel setup (only needed once, by the repo maintainer)

The `annotation/cloudflared/` directory holds `cert.pem`, `<tunnel-uuid>.json`
credentials, and `config.yml`. All three are gitignored — they're secrets.
If you're starting from a fresh clone, run these once:

```bash
mkdir -p annotation/cloudflared

# Log in — opens a browser, pick the your-domain zone, approve
docker run --rm -it \
  -v "$(pwd)/annotation/cloudflared:/home/nonroot/.cloudflared" \
  cloudflare/cloudflared:latest tunnel login

# Create the named tunnel
docker run --rm \
  -v "$(pwd)/annotation/cloudflared:/home/nonroot/.cloudflared" \
  cloudflare/cloudflared:latest tunnel create aksara-labeling

# Route the subdomain
docker run --rm \
  -v "$(pwd)/annotation/cloudflared:/home/nonroot/.cloudflared" \
  cloudflare/cloudflared:latest tunnel route dns aksara-labeling label.yourdomain.com
```

Then write `annotation/cloudflared/config.yml` with:

```yaml
tunnel: <UUID-from-tunnel-create>
credentials-file: /home/nonroot/.cloudflared/<UUID>.json

ingress:
  - hostname: label.yourdomain.com
    service: http://host.docker.internal:8080
  - service: http_status:404
```

After that, `docker compose up -d` just works on any machine with the
repo + `annotation/cloudflared/` files.

---

## 3. First-time account setup

1. Open **https://label.yourdomain.com** (or http://localhost:8080 if you're on the
   host machine)
2. Sign up with any email and password (stored locally in
   `annotation/ls_data/`, not sent anywhere)
3. You land on the projects page

---

## 4. Create the project

1. Click **Create Project**
2. **Project Name:** `Aksara Jawa Line OCR`
3. **Description:** `Line-level annotation of Aksara Jawa manuscripts`
4. Click **Labeling Setup**
5. Select **Custom template** (scroll to the bottom of templates)
6. Copy the entire contents of [`annotation/label_studio_config.xml`](../annotation/label_studio_config.xml) into the code box
7. Click **Save**

---

## 5. Import the eval-candidate pool

You're **not** annotating all 800 pages in `data/real/`. The eval set is a
deliberately stratified subset of **120 images** (80 Or. 1871 / 30 D Or. 15 /
10 Or. 1928), held back from the semi-synthetic background pool so model
evaluation is honest — the model never sees these pages during training, even
blurred.

The canonical list is **`data/eval_real_candidates.txt`** (committed to the
repo, deterministic via seed=42). Regenerate it with different targets / seed
by running:

```bash
uv run python scripts/pick_eval_candidates.py --seed 42
```

The same 120 filenames are passed to the synthetic generator's `--exclude`
flag, so training and eval are guaranteed disjoint.

### 5a. Cloudflare R2 cloud storage (recommended — default path)

Images live in the R2 bucket `aksara-jawa-images/real/eval_pool/`. Annotators'
browsers fetch images directly from R2 via Label Studio presigned URLs, so
tunnel bandwidth isn't consumed by image payloads — the tunnel only carries
UI + annotation JSON.

To seed the bucket (or resync after the candidate list changes), use the
project skill **`/annotation-r2-sync`** — it reads `data/eval_real_candidates.txt`,
loads R2 credentials from `annotation/.env`, and uploads only those 120 files
via `rclone copy --files-from`.

Configure the Label Studio source storage once per project:

1. Project → **Settings** → **Cloud Storage** → **Add Source Storage**

| Field | Value |
|---|---|
| Storage Type | `AWS S3` |
| Bucket Name | `aksara-jawa-images` |
| Bucket Prefix | `real/eval_pool/` |
| File Filter Regex | `.*\.jpg$` |
| Region Name | `auto` |
| S3 Endpoint | value of `R2_ENDPOINT` from `annotation/.env` |
| Access Key ID / Secret Access Key | from `annotation/.env` |
| Treat every bucket object as a source file | **ON** |
| Use pre-signed URLs | **ON** (this is the whole point) |
| Presigned URL TTL | `3600` |

2. Click **Add Storage → Sync Storage**. Exactly 120 tasks should appear.

**CORS caveat:** the R2 bucket must allow `GET`/`HEAD` from the public tunnel
hostname (e.g. `https://label.ketok.id`) or image fetches fail silently in the
browser. Set via R2 dashboard → bucket → Settings → CORS Policy. Template in
`annotation/.env.example`.

**Task-count sanity check:** after Sync Storage completes, open Data Manager.
If you see more than 120 tasks, delete the orphans — they're leftovers from
an older `Local files` source (§5b) or drag-and-drop import (§5c).

### 5b. Local file storage (fallback — no R2 configured)

If R2 isn't set up yet (initial bootstrap, air-gapped machine, etc.), point
Label Studio at the bind-mounted `data/real/`:

1. Project → **Settings** → **Cloud Storage** → **Add Source Storage**
2. **Storage Type:** `Local files`
3. **Absolute local path:** `/label-studio/files/data/real`
4. **File Filter Regex:** `.*\.(jpg|jpeg|png)$`
5. **Treat every bucket object as a source file:** ON
6. Click **Add Storage** → **Sync Storage**

This imports **all 800 images** — the candidate-list restriction isn't
enforced at this layer. You'll need to manually filter the Data Manager to
annotate only filenames listed in `data/eval_real_candidates.txt`; or move
those 120 files into a `data/real/eval_pool/` subdirectory and point the
storage source there. The R2 path (§5a) avoids all of this by using a
bucket prefix.

### 5c. Drag-and-drop (small one-offs only)

1. Go to project → **Import**
2. Drag image files into the drop zone
3. Click **Import**

Files get copied into `annotation/ls_data/` rather than read from `data/real/`
or R2, so provenance via `sources.csv` is lost. Prefer 5a.

---

## 6. Annotate

Keyboard shortcuts speed this up significantly:

| Key | Action |
|---|---|
| `1` | Select `line` label |
| `2` | Select `illegible` label |
| Drag on image | Draw bounding box |
| `Tab` | Move focus to transcription box |
| `Ctrl+Enter` | Submit and load next task |
| `Ctrl+Z` | Undo |
| Scroll wheel | Zoom |

### Annotation rules

1. **One box per LINE** — not per word, not per character.
2. **Tight bounding boxes** — include the full line height (including sandhangan above
   and pasangan below), no extra whitespace.
3. **Overlapping boxes between adjacent lines are EXPECTED, not a bug.**
   On dense manuscripts the sandhangan-above of line N+1 and the pasangan-below of
   line N physically share vertical pixel space. Draw both boxes to fully contain
   their own glyphs — overlaps in Y are fine.
   - **Line assignment rule:** a glyph belongs to whichever line its **base consonant**
     sits on. All of its sandhangan, pasangan, and combining marks go with that base
     into the same line's box + transcription.
   - **A box may visually contain pixels from the neighbour line.** That's OK — the box
     is a spatial region, not a pixel partition. Your transcription for that box
     contains **only** the characters belonging to this line. Do not re-transcribe
     glyphs that belong to the neighbour, even if they're visible inside your box.
   - **When the page is visibly tilted** (common on palm-leaf and old European-paper
     manuscripts), switch from rectangle to polygon/quadrilateral tool — rectangles
     waste space on tilted lines and swallow too much neighbour content.
4. **Transcribe in Unicode Javanese** — the U+A980–U+A9DF block. Do **not** use Latin
   transliteration (no `hanacaraka`, only `ꦲꦤꦕꦫꦏ`).
5. **Preserve punctuation** — pada lingsa `꧈`, pada lungsi `꧉`, and other pada marks
   must appear in the transcription exactly as in the image.
6. **Mark illegible lines** with the `illegible` label. Leave transcription empty.
   These are excluded from training automatically. Use `illegible` for:
   - binding-gutter regions where adjacent lines physically merge into one stroke
   - ink bleed-through from the opposite leaf where glyph ownership is unclear
   - smudged base consonants whose line assignment can't be determined
   - non-Aksara-Jawa content that slipped into the page (Arabic insertions on the
     Bonang item, Latin catalogue stamps, library ex-libris marks)
7. **Script type tag** (optional but valuable) — pick `printed`, `handwritten`, or
   `manuscript` per region for later stratified evaluation.

### Input methods for Aksara Jawa Unicode

- **Aksara Jawa keyboard layout:** https://www.branah.com/javanese
  Type phonetically, copy the Unicode output into Label Studio.
- **System keyboard (macOS):** System Settings → Keyboard → Input Sources →
  add Javanese.
- **Fallback:** Google Translate does not support Aksara Jawa, but
  [aksarajawa.com](https://aksarajawa.com) provides Latin → Unicode conversion.

---

## 7. Export annotations

1. Project → **Export**
2. Format: **JSON** (not JSON-MIN, we need the full structure)
3. Save the file as `annotation/export.json`

---

## 8. Convert to PaddleOCR + ERNIEKit format

From the repository root:

```bash
uv run python scripts/labelstudio_to_paddleocr.py \
  --input annotation/export.json \
  --image_dir data/real/ \
  --label_out data/real/Label.txt \
  --jsonl_out data/real/ground_truth.jsonl
```

This produces:

- `data/real/Label.txt` — PaddleOCR detection format with pixel bounding boxes
- `data/real/ground_truth.jsonl` — conversation format (one line per image, all lines joined)

To merge real data into the training set for fine-tuning:

```bash
uv run python scripts/convert_format.py \
  --input data/synthetic/ground_truth.jsonl data/real/ground_truth.jsonl \
  --image_dir data/synthetic/ data/real/ \
  --output training/ocr_vl_sft-train_aksara_jawa.jsonl \
  --shuffle
```

---

## 9. Quality checklist before committing

Run these sanity checks before declaring a batch done:

- [ ] No empty transcriptions on `line` regions
- [ ] All `illegible` regions have no transcription
- [ ] No Latin characters in transcription strings (run `grep -P '[a-zA-Z]' data/real/Label.txt` — should be empty)
- [ ] Bounding boxes have non-zero width and height
- [ ] Image filenames in `Label.txt` exist in `data/real/`

A 10-image spot-check by a second reader catches most errors. For the eval set,
every image should be reviewed twice.

---

## 10. Target counts for the competition eval set

Current Leiden-only pool (all public domain, redistributable):

| Source | Candidate pool | Target clean annotations | `script_type` tag |
|---|---|---|---|
| Or. 1871 (Panji romance, European paper) | 80 | ~65 | manuscript |
| D Or. 15 (Babad Paku Alaman, illuminated court) | 30 | ~25 | manuscript |
| Or. 1928 (Kitab pangeran Bonang, dluwang) | 10 | ~8 | manuscript |
| **Total** | **120** | **~100 clean** | — |

120 images × ~83% = ~100 usable — the gap is a 20% rejection budget for
damaged pages, non-Aksara-Jawa lines (Arabic on the Bonang item is common),
and binding/flyleaves that slipped past the collector's
`--skip-canvases-per-manifest` heuristic. Mark those as `illegible`; the
converter (§8) drops them.

Future supplementary eval sources — not yet integrated, pool-building work
pending per `docs/data_report.md` §2.2:

- PNRI Khastara printed books (~30, printed tier)
- Yogyakarta street signage photos (~20, printed tier)
- EAP palm-leaf manuscripts (~20, handwritten / palm-leaf manuscript tier)

Aim for minimum 50, target 100. **Synthetic images never belong in the eval set.**
