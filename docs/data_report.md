# Training Data Construction Report
## PaddleOCR-VL Aksara Jawa OCR Fine-tuning

**Project:** PaddleOCR Global Derivative Model Challenge (Hackathon 10th)  
**Task:** Aksara Jawa (Javanese Script) OCR & Transcription  
**GitHub:** https://github.com/setilanaji/paddleocr-aksara-jawa  
**HuggingFace Dataset:** https://huggingface.co/datasets/setilanaji/aksara-jawa-ocr  
**Team:** Yudha Setyaji, Budi Haryono  
**Date:** April 2026

---

## 1. Task Description

Aksara Jawa (ꦲꦏ꧀ꦱꦫꦗꦮ), also known as Carakan or Hanacaraka, is the traditional script of the Javanese language. It is an abugida (alphasyllabary) written left-to-right, used since the 15th century across Central Java, Yogyakarta, and East Java, Indonesia.

The Unicode block U+A980–U+A9DF contains 91 characters covering:

| Component | Description | Count |
|---|---|---|
| Carakan | Base consonant characters | 20 |
| Sandhangan | Vowel diacritical marks | 11 |
| Pasangan | Stacked consonant forms (consonant clusters) | 20 |
| Pada | Punctuation marks | 8 |
| Digits | Javanese numerals | 10 |
| Other | Special marks and modifiers | 22 |

The OCR task is defined as: given an image containing Aksara Jawa text, transcribe the text into the correct Unicode string. This is a sequence-to-sequence vision-language task with high structural complexity due to the stacked pasangan forms and combining sandhangan marks.

**Why this scenario is scarce:** As of April 2026, no publicly available fine-tuned OCR model exists for Aksara Jawa. Google Translate does not support Aksara Jawa input. The closest existing tools are rule-based transliteration utilities that only handle clean digital text, not image-based recognition.

---

## 2. Data Sources

### 2.1 Synthetic Data (Primary — Open Source)

**Source:** Noto Sans Javanese font rendered programmatically using Python Pillow  
**Font:** NotoSansJavanese-Regular.ttf  
**Font License:** SIL Open Font License 1.1 (fully open source, commercial use permitted)  
**Font Coverage:** 405 glyphs, full Unicode Javanese block U+A980–U+A9DF  
**Redistribution:** Permitted under SIL OFL 1.1

Synthetic images were generated using `scripts/generate_aksara.py`, a custom pipeline that renders Javanese Unicode text from a curated corpus onto randomised backgrounds with augmentation.

### 2.2 Real Manuscript Data (Supplementary — Training Only)

**Source:** PNRI Khastara (Khasanah Pustaka Nusantara)  
**URL:** https://khastara.perpusnas.go.id  
**Collection:** Digitised ancient Javanese manuscripts from the National Library of Indonesia  
**Access:** Open access — publicly downloadable by anyone  
**License:** Open access for research and educational purposes  
**Redistribution:** Not permitted — kept in private `data/real/` folder, excluded from public repository

**Source:** British Library Endangered Archives Programme (EAP)  
**URL:** https://eap.bl.uk  
**Relevant projects:** EAP280 (Old Javanese palm-leaf manuscripts), EAP1268 (North Coast Javanese manuscripts)  
**Access:** Free online access  
**License:** Research purposes only  
**Redistribution:** Not permitted — kept in private training folder only

---

## 3. Text Corpus Construction

A 303-line Javanese Unicode corpus was compiled covering:

- Everyday Javanese vocabulary and phrases
- Cultural and ceremonial language
- Nature, geography, and place names
- Government and administrative terminology
- Educational and academic vocabulary
- Proverbs and traditional expressions

**Corpus statistics:**

| Metric | Value |
|---|---|
| Total unique sentences | 303 |
| Average characters per sentence | ~24 |
| Minimum sentence length | 8 characters |
| Maximum sentence length | 47 characters |
| Unicode block covered | U+A980–U+A9DF (primary) |
| Unique Unicode code points | 68 distinct Aksara Jawa characters |

The corpus was designed to maximise character combination diversity — particularly ensuring coverage of all 20 pasangan (stacked consonant forms) and all major sandhangan (vowel marks), since these are the most visually complex elements of Aksara Jawa and the most likely sources of OCR errors.

---

## 4. Synthetic Image Generation Pipeline

### 4.1 Single-line Mode

Each image contains one sentence rendered as a horizontal text strip.

| Parameter | Values |
|---|---|
| Image width | 400, 512, 640, 800 px (random) |
| Image height | 64, 80, 96, 112, 128 px (random) |
| Font size | 18, 20, 22, 24, 26, 28, 32 pt (random) |
| Text position | Random within safe margins |

### 4.2 Multi-line Mode

Each image contains 2–5 sentences stacked vertically as a paragraph block, simulating real-world manuscript layout.

| Parameter | Values |
|---|---|
| Image width | 512, 640, 800, 960 px (random) |
| Image height | 200, 256, 320, 400, 480 px (random) |
| Font size | 16, 18, 20, 22, 24 pt (random) |
| Lines per image | 2–5 (random) |
| Line spacing | 1.6× line height |
| Left margin | 16–48 px (random) |
| Top margin | 12–32 px (random) |

### 4.3 Background Variation

Eight background tones simulate different paper conditions:

| Tone | RGB | Description |
|---|---|---|
| Clean white | (255, 255, 255) | Modern printed text |
| Warm off-white | (252, 248, 240) | Lightly aged paper |
| Aged paper | (245, 240, 225) | Older document |
| Old manuscript | (240, 235, 215) | Antique paper |
| Cream | (250, 245, 235) | Standard cream |
| Antique | (235, 230, 210) | Heavily aged |
| Warm cream | (248, 244, 230) | Warm-toned paper |
| Parchment | (242, 237, 220) | Parchment simulation |

### 4.4 Text Colour Variation

Six text colours simulate different ink conditions:

| RGB | Description |
|---|---|
| (10, 10, 10) | Near-black — modern ink |
| (30, 25, 20) | Dark brown-black — aged ink |
| (20, 20, 40) | Dark navy — blue-black ink |
| (40, 25, 10) | Dark sepia — sepia ink |
| (15, 15, 15) | Soft black — faded modern ink |
| (25, 20, 30) | Dark purple-black — historical ink |

---

## 5. Augmentation Strategy

Three severity levels simulate real-world image capture conditions:

### Light Augmentation (eval set + 40% of training)

| Transform | Parameter |
|---|---|
| Rotation | ±2° |
| Gaussian blur | p=0.20, radius 0.3–0.8 |
| Brightness | 0.70–1.30× |
| Contrast | 0.80–1.25× |
| Noise | p=0.15, ±3–8 intensity |
| JPEG compression | p=0.50, quality 60–95 |

### Medium Augmentation (40% of training)

| Transform | Parameter |
|---|---|
| Rotation | ±5° |
| Gaussian blur | p=0.40, radius 0.3–1.5 |
| Brightness | 0.70–1.30× |
| Contrast | 0.80–1.25× |
| Noise | p=0.45, ±3–15 intensity |
| JPEG compression | p=0.50, quality 60–95 |

### Heavy Augmentation (20% of training)

| Transform | Parameter |
|---|---|
| Rotation | ±10° |
| Gaussian blur | p=0.65, radius 0.3–2.5 |
| Brightness | 0.70–1.30× |
| Contrast | 0.80–1.25× |
| Noise | p=0.75, ±3–25 intensity |
| JPEG compression | p=0.50, quality 60–95 |

---

## 6. Dataset Statistics

### 6.1 Training Set (three-tier composition)

Training data is split into three tiers, each targeting a different
generalization axis:

| Tier | Images | Source | Purpose |
|---|---|---|---|
| **Pure synthetic** | 2,000 | Noto Sans Javanese (Regular + Bold), solid/aged-paper backgrounds | Core script coverage, typography variety |
| **Semi-synthetic** | 1,000 | Same text rendering on **real blurred Dreamsea manuscript backgrounds** | Domain adaptation: teaches the model real paper texture, ink tone, bleed-through without needing transcription |
| **Real (unannotated pool)** | 134 | Dreamsea IIIF-served Javanese manuscripts (via `scripts/dreamsea_collect.py`) | Triage + annotation source for eval set and future supplementary training |
| **Total trainable** | **3,000** | — | Merged + shuffled into `training/ocr_vl_sft-train_aksara_jawa.jsonl` |

Pure-synthetic breakdown (seed=42, `scripts/generate_aksara.py`):

| Metric | Value |
|---|---|
| Single-line / multi-line split | 1,191 single (59.6%) / 809 multi (40.4%) |
| Font distribution | NotoSansJavanese-Regular.ttf + NotoSansJavanese-Bold.ttf (uniform random per image) |
| Pasangan-stress ratio | 20% (sampled from `assets/corpus_jawa_pasangan.txt`, 150 lines, avg 6 pasangan clusters/line) |
| Augmentation distribution | 771 light / 830 medium / 399 heavy |

Semi-synthetic breakdown (seed=142, `--real_bg_ratio 1.0`):

| Metric | Value |
|---|---|
| Total images | 1,000 |
| Real backgrounds pool | 134 Dreamsea IIIF pages (filtered by `--script Javanese`) |
| Background treatment | Random crop at target image size → Gaussian blur (σ=8–16) → brightness jitter (0.85–1.15) |
| Pasangan-stress ratio | 20% (same corpus as pure-synthetic) |
| Augmentation distribution | 403 light / 393 medium / 204 heavy |

### 6.2 Evaluation Set

| Metric | Value | Notes |
|---|---|---|
| Total images (v1, synthetic) | 150 | Reproducible via `generate_aksara.py --eval --seed 42` |
| Augmentation | Light only | Preserves legibility for accurate CER/WER scoring |
| Random seed | 42 | — |
| Target v2 (in progress) | 50–100 real images | Annotated at line level with Unicode Javanese transcription + `script_type` tag (printed/handwritten/manuscript) via Label Studio; see `docs/annotation_guide.md` |
| Tiered split target | ~30 printed + ~30 manuscript + ~20 handwritten + ~20 signage | For per-tier CER reporting via `scripts/evaluate.py` |

### 6.3 Aksara Jawa Character Coverage

| Feature | Coverage | Evidence |
|---|---|---|
| Carakan (base consonants) | All 20 covered | In main corpus + all 20 used as stacked second position in pasangan-stress corpus |
| Sandhangan (vowel marks) | All major marks covered | wulu ꦶ, suku ꦸ, pepet ꦼ, taling ꦺ, taling-tarung ꦺꦴ |
| Pasangan (stacked forms) | Dense — every pasangan-stress line contains ≥1 cluster | Avg 6.09 pasangan clusters per stress-corpus line |
| Pada (punctuation) | pada lingsa ꧈, pada lungsi ꧉ | Present in corpus + stress-corpus |

### 6.4 Real-Data Sources (Provenance)

All 134 real-manuscript images used for semi-synthetic backgrounds are sourced
from Dreamsea via the vhmml.org IIIF Image API. Provenance is tracked
per-image in `data/real/sources.csv` with columns:

```
local_filename, source_url, manifest_url, canvas_label
```

Filter used for harvest (via `scripts/dreamsea_collect.py`):

```
--script Javanese --country Indonesia --limit 30
```

This returned 30 manifests × up to 5 pages each. Three manifests had fewer
than 5 canvases (palm-leaf single-leaf fragments), giving 134 final pages
(24 × 5 + 4 + 3·2 + 1·2).

---

## 7. Annotation Format

### 7.1 PaddleOCR Detection Format (Label.txt)

One line per image. Used for text detection training.

```
aksara_0001.jpg	[{"transcription": "ꦲꦤꦏ꧀ꦏꦶ", "points": [[45,12],[207,12],[207,45],[45,45]], "label": "aksara_jawa", "illegibility": false}]
```

Fields:
- `transcription` — Unicode Aksara Jawa ground truth string
- `points` — four-point bounding box `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]`
- `label` — always `aksara_jawa`
- `illegibility` — always `false` for synthetic data

### 7.2 VLM Conversation Format (ground_truth.jsonl)

One JSON object per line. Used for evaluation and reference.

```json
{
  "image": "aksara_0001.jpg",
  "conversations": [
    {"role": "user", "content": "Baca teks Aksara Jawa dalam gambar ini."},
    {"role": "assistant", "content": "ꦲꦤꦏ꧀ꦏꦶꦱꦼꦩꦸꦮꦃ"}
  ]
}
```

### 7.3 PaddleFormers SFT Messages Format (ocr_vl_sft.jsonl)

Required format for `paddleformers-cli train` when
`train_dataset_type: messages` is set in the config (our case). Generated
directly by the generator with the `--erniekit` flag, or via
`scripts/convert_format.py`. Upstream reference: [PaddleFormers PaddleOCR-VL
best practice](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/best_practices/PaddleOCR-VL).

```json
{
  "messages": [
    {"role": "user", "content": "<image>OCR:"},
    {"role": "assistant", "content": "ꦲꦤꦏ꧀ꦏꦶꦱꦼꦩꦸꦮꦃ"}
  ],
  "images": ["./data/synthetic/aksara_0001.jpg"]
}
```

The `<image>` placeholder in the user turn marks where the image token is
inserted; the assistant turn is the prediction target.

---

## 8. Quality Control

### 8.1 Synthetic Data

- All ground truth labels are programmatically verified — the generator knows exactly what text was rendered
- Bounding boxes are computed from Pillow `textbbox()` — pixel-accurate
- Ground truth sanity check: CER=0.00%, WER=0.00%, exact match 150/150 (verified by `evaluate.py --gt_only`)
- Unicode NFC normalisation applied to both corpus input and CER/WER comparison (both sides)
- Empty-annotation fallback removed — images where text doesn't fit are regenerated with smaller fonts (up to 5 retries) rather than written with placeholder illegibility boxes that would poison training

### 8.2 Real Data Annotation (Live pipeline)

Real manuscript images from Dreamsea / Khastara are annotated in **Label Studio**,
self-hosted at the team's private URL via a Cloudflare Access-gated tunnel.
Full workflow in [`annotation_guide.md`](annotation_guide.md).

Annotation contract (enforced by `annotation/label_studio_config.xml`):

1. **One bounding box per text LINE** — not word, not character
2. **Unicode Javanese transcription only** — no Latin transliteration allowed
3. **Preserve pada punctuation** — pada lingsa ꧈, pada lungsi ꧉ are transcribed literally
4. Damaged / unreadable lines get the `illegible` label and empty transcription (excluded from training automatically by `scripts/labelstudio_to_paddleocr.py`)
5. **Script type tag** per region — `printed` / `handwritten` / `manuscript` — enables stratified evaluation via `scripts/evaluate.py` per-tier CER

Pre-annotation triage:

- `scripts/build_triage_page.py` generates a static HTML grid of thumbnails with keep/drop checkboxes (localStorage-backed for resume); the corresponding `scripts/apply_triage.py` deletes dropped images + cleans `sources.csv` rows
- Removes false positives from the Dreamsea "script=Javanese" filter (occasional non-Aksara-Jawa pages)

### 8.3 Inter-Annotator Agreement

Two annotators independently transcribe the same 20-image sample. Pairwise CER
between their two Unicode outputs, averaged across the sample, is the
agreement number.

Computed by `scripts/annotator_agreement.py` from two Label Studio JSON exports:

```bash
uv run python scripts/annotator_agreement.py \
  --annotator_a annotation/export_yudha.json \
  --annotator_b annotation/export_budi.json
```

The script reports per-image CER, mean CER across shared images, and the count
of shared images. CER ≤ 0.05 on transcription is considered strong agreement
for a novel script with no prior OCR dataset.

### 8.4 Second-Pass Review

Every annotation that ends up in the evaluation set is reviewed by a second
reader before acceptance. Rejection reasons tracked: misread glyph,
missed sandhangan, wrong pasangan, incorrect pada punctuation, Latin
transliteration leaked in. Rejected items go back to the first annotator for
correction.

---

## 9. Fine-tuning Configuration

Training uses ERNIEKit on RunPod (A100 40GB Community Cloud). See [runpod_setup.md](runpod_setup.md) for the full launch + install recipe.

| Parameter | Value |
|---|---|
| Base model | `PaddlePaddle/PaddleOCR-VL` (v1.0) |
| Fine-tuning method | LoRA (Supervised Fine-Tuning) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Epochs | 5 |
| Config | `training/aksara_jawa_lora_config.yaml` (version-controlled) |
| Platform | RunPod, A100 40GB |

Training command (config is version-controlled in the repo at `training/aksara_jawa_lora_config.yaml`):

```bash
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train training/aksara_jawa_lora_config.yaml
```

The config snapshots the exact hyperparameters, dataset paths, and model identifier used for this run. The upstream reference (`training/paddleocr-vl_lora_16k_config.yaml`) is preserved unchanged in the same directory so reviewers can diff against the project-specific adaptations.

---

## 10. Known Limitations

1. **Corpus size** — 303 unique sentences provides adequate coverage for initial fine-tuning but limited vocabulary breadth. Expansion to 1000+ sentences is planned.

2. **Pasangan coverage** — Stacked consonant forms are underrepresented in the current corpus. These are among the most visually complex characters and a known source of OCR errors.

3. **Single font** — Only Noto Sans Javanese is used for synthetic generation. Real-world Aksara Jawa appears in multiple font styles and historical letterform variants.

4. **No handwriting** — Current dataset contains only printed/typeset Aksara Jawa. Handwritten manuscript recognition requires separate data.

5. **Synthetic-only public set** — Real manuscript images are used for training but cannot be redistributed. The public dataset is synthetic only.

---

## 11. Reproducibility

The complete synthetic dataset can be reproduced from scratch:

```bash
git clone git@github.com:setilanaji/paddleocr-aksara-jawa.git
cd paddleocr-aksara-jawa
uv sync

curl -L -o assets/NotoSansJavanese-Regular.ttf \
  "https://github.com/notofonts/javanese/raw/main/fonts/ttf/NotoSansJavanese-Regular.ttf"

# Training set
uv run python scripts/generate_aksara.py \
  --count 1000 --output data/synthetic/ --seed 42 \
  --erniekit --image_dir ./data/synthetic

# Eval set
uv run python scripts/generate_aksara.py \
  --count 150 --output data/eval/ --eval \
  --erniekit --image_dir ./data/eval

# Convert to ERNIEKit training format
uv run python scripts/convert_format.py \
  --input data/synthetic/ground_truth.jsonl \
  --image_dir data/synthetic/ \
  --output training/ocr_vl_sft-train_aksara_jawa.jsonl
```

All random operations use `seed=42` for deterministic output.

---

## 12. Future Work

- Expand corpus to 1000+ unique sentences from Javanese literature and Wikipedia
- Add multi-font synthetic generation
- Integrate real manuscript images from PNRI Khastara after annotation
- Add handwriting augmentation
- Build a dedicated pasangan test set to measure stacked consonant accuracy separately
- Collect real-world signage photos from Yogyakarta and Surakarta

---

*Report generated: April 2026*  
*Authors: Yudha Setyaji, Budi Haryono*  
*License: Apache-2.0*