# Project Status — 2026-04-23

Session handoff for the PaddleOCR Aksara Jawa fine-tuning project.
Preliminary deadline: **2026-05-29** · Round 2 leaderboard: 2026-05-11.

## What's running right now

| Process | State | Notes |
|---|---|---|
| `aksara-labelstudio` (Docker) | running | Label Studio 1.13.1 on `http://localhost:8080` |
| `aksara-cloudflared` (Docker) | running | Tunnel `aksara-labeling` → `https://label.yourdomain.com` |
| Dreamsea harvest | background task `bkset9etg` | 30 manuscripts × 5 pages @ 1200px, ETA ~45 min, slow due to vhmml.org IIIF scaling |

Compose stack: `docker compose -f annotation/docker-compose.yml up -d`

## Infrastructure set up this session

- **Label Studio + Cloudflare tunnel** — `annotation/docker-compose.yml` runs both. Data persists in `annotation/ls_data/` (gitignored). Tunnel credentials in `annotation/cloudflared/` (gitignored). Public hostname configured via `annotation/.env` (gitignored; see `annotation/.env.example`).
- **Cloudflare Access policy** — `label.yourdomain.com` is email-gated. Only the emails in the Zero Trust Access application can sign up on Label Studio.
- **Annotation config** — `annotation/label_studio_config.xml` defines line + illegible labels, per-region transcription textbox, optional script-type tag.
- **Annotation guide** — `docs/annotation_guide.md` has full step-by-step for Budi or anyone joining.

## Scripts added this session

| Script | Purpose |
|---|---|
| `scripts/collect_manuscripts.py` | Generic IIIF v2/v3 and direct-URL image downloader → `data/real/` |
| `scripts/labelstudio_to_paddleocr.py` | Label Studio JSON export → `Label.txt` + `ground_truth.jsonl` |
| `scripts/khastara_collect.py` | Khastara API search → `annotation/khastara_shopping_list.csv` (URLs for manual browser download; Khastara blocks programmatic access via Cloudflare) |
| `scripts/split_pdfs.py` | Local PDF folder → `data/real/` page JPGs (for Khastara PDFs once downloaded manually) |
| `scripts/dreamsea_collect.py` | Dreamsea search → IIIF manifest URL list (feeds `collect_manuscripts.py`) |

## Demo ready for push

`demo/` directory is complete and HF-Space-ready:
- `demo/app.py` — Gradio app, transformers-based, ZeroGPU-decorated
- `demo/requirements.txt`
- `demo/README.md` — HF Space YAML frontmatter
- `demo/examples/` — 5 sample images

**Default MODEL_PATH is `PaddlePaddle/PaddleOCR-VL`** (base). After the fine-tune publishes to `setilanaji/PaddleOCR-VL-Aksara-Jawa`, set `MODEL_PATH` in Space settings.

**Push is deferred** per user instruction (2026-04-23). See Task #10.

## Data state

- `data/synthetic/` — 1,000 generated images, reproducible from seed 42
- `data/eval/` — 150 synthetic eval images (⚠️ must be replaced with real before submission — synthetic eval = one-vote veto per competition rules)
- `data/real/` — starting to populate via Dreamsea harvest (background)

Provenance is tracked in `data/real/sources.csv` with columns: `local_filename, source_url, manifest_url, canvas_label`.

## Source-by-source download status

| Source | Scripted access | Notes |
|---|---|---|
| Dreamsea (vhmml.org IIIF) | fully automated | `dreamsea_collect.py` + `collect_manuscripts.py` — works end-to-end. Caveat: Dreamsea's language=javanese filter returns manuscripts where *any* item is in Javanese, so many results are Arabic-script with a Javanese colophon. Needs visual triage (Task #2). |
| Khastara (PNRI) | metadata only | API search works. File downloads blocked by Cloudflare on `file-opac.perpusnas.go.id`. Workflow: `khastara_collect.py` emits shopping-list CSV → human downloads PDFs via browser → `split_pdfs.py` extracts pages. |
| EAP / Wikipedia / signage | supported via generic downloader | Paste IIIF manifest URLs into `annotation/manifests.txt` or direct URLs into `annotation/image_urls.txt`, run `collect_manuscripts.py`. |

## Competition scoring map (6 dimensions, 100 pts total)

| Dimension | Max | Current state | Gap |
|---|---|---|---|
| Evaluation Set Quality | 20 | 0 (synthetic → veto) | Annotate 50–100 real images. Tasks #2-5 |
| Scenario Scarcity | 15 | near-max likely | Aksara Jawa genuinely has no public OCR model |
| Task Complexity | 15 | near-max likely | Pasangan + sandhangan + stacking is real complexity |
| Training Dataset Rigor | 20 | ~12 estimated | Add inter-annotator agreement (Task #6), expand `data_report.md` (Task #11) |
| Fine-tuning Strategy | 10 | 0 | Run fine-tune (Task #7) + ablations (Task #8) |
| Documentation & Open Source | 20 | ~15 estimated | Publish Gradio demo (Task #10) is the biggest single win (5 pts) |

Top 10 threshold: ≥60 overall. Independent "High-Quality Eval Set Award" (1,000 RMB): ≥12/20 on the Evaluation Set Quality dimension alone.

## Submission targets

- **Round 2 leaderboard** (May 11): submit partial materials by ~May 7 for mentor-feedback email. See Task #12.
- **Preliminary deadline** (May 29): final eval set + training report + GitHub + HF model. Can iterate up to deadline; last submission wins.
- **Submission email**: `ext_paddle_oss@baidu.com` + `paddleocr@baidu.com` + `cuicheng01@baidu.com` + `liujiaxuan01@baidu.com` · Subject: `PaddleOCR Derivative Model Challenge - [Material Name] - setilanaji`

## Critical path

Annotation → export → fine-tune → evaluate → demo → submit. Annotation is the single biggest bottleneck — everything downstream waits on it.

## Active task list

See `TaskList` (12 tasks created 2026-04-23). Critical-path order: #1 → #2 → #3 → #4 → #5 → #7 → #9 → #11 → #12. Parallel tracks: #6 (annotator agreement), #8 (ablations after #7), #10 (demo any time).
