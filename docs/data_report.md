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

### 6.1 Training Set

| Metric | Value |
|---|---|
| Total images | 1,000 |
| Single-line images | ~591 (59.1%) |
| Multi-line images | ~409 (40.9%) |
| Light augmentation | ~400 images |
| Medium augmentation | ~400 images |
| Heavy augmentation | ~200 images |
| Random seed | 42 (reproducible) |
| Total annotation boxes | ~2,200 |

### 6.2 Evaluation Set

| Metric | Value |
|---|---|
| Total images | 150 |
| Single-line images | 150 (100%) |
| Augmentation | Light only |
| Random seed | 42 (reproducible) |
| Total annotation boxes | 150 |

### 6.3 Aksara Jawa Character Coverage

| Feature | Coverage |
|---|---|
| Carakan (base consonants) | All 20 covered |
| Sandhangan (vowel marks) | All major marks covered |
| Pasangan (stacked forms) | Partial — corpus-dependent |
| Pada (punctuation) | Basic coverage |

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

### 7.3 ERNIEKit SFT Format (ocr_vl_sft.jsonl)

Required format for `erniekit train`. Generated directly by the generator with `--erniekit` flag, or via `scripts/convert_format.py`.

```json
{
  "image_info": [{"matched_text_index": 0, "image_url": "./data/synthetic/aksara_0001.jpg"}],
  "text_info": [
    {"text": "OCR:", "tag": "mask"},
    {"text": "ꦲꦤꦏ꧀ꦏꦶꦱꦼꦩꦸꦮꦃ", "tag": "no_mask"}
  ]
}
```

The `"tag": "mask"` marks the prompt as input, `"tag": "no_mask"` marks the transcription as the prediction target.

---

## 8. Quality Control

### 8.1 Synthetic Data

- All ground truth labels are programmatically verified — the generator knows exactly what text was rendered
- Bounding boxes are computed from Pillow `textbbox()` — pixel-accurate
- Ground truth sanity check: CER=0.00%, WER=0.00%, exact match 150/150 (verified by `evaluate.py --gt_only`)
- Unicode normalisation applied to all corpus text before rendering

### 8.2 Real Data Annotation (Planned)

Real manuscript images from Khastara will be annotated using PPOCRLabel:

1. Auto-detection of text regions
2. Manual correction of detected boundaries
3. Manual transcription of each region into Unicode Aksara Jawa
4. Verification by a Javanese language reader
5. Second-pass review before inclusion in training set

---

## 9. Fine-tuning Configuration

Training uses ERNIEKit on Baidu AI Studio (V100 16GB).

| Parameter | Value |
|---|---|
| Base model | `PaddlePaddle/PaddleOCR-VL` (v1.0) |
| Fine-tuning method | LoRA (Supervised Fine-Tuning) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Epochs | 5 |
| Config | `run_ocr_vl_sft_16k.yaml` |
| Platform | Baidu AI Studio, V100 16GB |

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 \
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  model_name_or_path=PaddlePaddle/PaddleOCR-VL \
  train_dataset_path=./training/ocr_vl_sft-train_aksara_jawa.jsonl
```

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