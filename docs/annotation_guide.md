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

## 5. Import images

### 5a. Local file storage sync (recommended)

Images downloaded by `scripts/collect_manuscripts.py` / `scripts/split_pdfs.py`
land in `data/real/`, which the compose stack mounts read-only inside the
container. Point Label Studio at it:

1. Project → **Settings** → **Cloud Storage** → **Add Source Storage**
2. **Storage Type:** `Local files`
3. **Absolute local path:** `/label-studio/files/data/real`
4. **File Filter Regex:** `.*\.(jpg|jpeg|png)$`
5. **Treat every bucket object as a source file:** ON
6. Click **Add Storage** → **Sync Storage**

Any new images added to `data/real/` later show up after another **Sync Storage** click.

### 5b. Drag-and-drop (small one-offs only)

1. Go to project → **Import**
2. Drag image files into the drop zone
3. Click **Import**

Up to ~100 images per batch; files get copied into `annotation/ls_data/`
rather than read from `data/real/`, so provenance via `sources.csv` is lost.
Prefer 5a.

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
3. **Transcribe in Unicode Javanese** — the U+A980–U+A9DF block. Do **not** use Latin
   transliteration (no `hanacaraka`, only `ꦲꦤꦕꦫꦏ`).
4. **Preserve punctuation** — pada lingsa `꧈`, pada lungsi `꧉`, and other pada marks
   must appear in the transcription exactly as in the image.
5. **Mark illegible lines** with the `illegible` label. Leave transcription empty.
   These are excluded from training automatically.
6. **Script type tag** (optional but valuable) — pick `printed`, `handwritten`, or
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

| Source | Images | Script type |
|---|---|---|
| PNRI Khastara (printed Javanese books) | ~30 | printed |
| PNRI Khastara (handwritten manuscripts) | ~30 | manuscript |
| Yogyakarta street signage photos | ~20 | printed |
| EAP / Dreamsea palm-leaf manuscripts | ~20 | manuscript |
| **Total** | **~100** | |

Aim for minimum 50, target 100. Synthetic images never belong in the eval set.
