# PaddleOCR-VL Aksara Jawa

Fine-tuning [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) on Aksara Jawa (Javanese script) OCR and transcription, submitted to the [PaddleOCR Global Derivative Model Challenge — Hackathon 10th](https://github.com/PaddlePaddle/PaddleOCR/issues/17858).

## Background

Aksara Jawa (ꦲꦏ꧀ꦱꦫꦗꦮ), also known as Carakan or Hanacaraka, is the traditional abugida script of the Javanese language. It has been in active use since the 15th century across Central Java, Yogyakarta, and East Java, and remains part of the official Indonesian school curriculum in those regions. The script appears on government signage, street signs, museum labels, batik patterns, and tens of thousands of historical manuscripts held in national and university libraries.

Despite its cultural significance and active institutional use, no publicly available fine-tuned OCR model exists for Aksara Jawa. Google Translate does not support the script. Existing tools are limited to rule-based transliteration of clean digital text and cannot handle image-based recognition. This project addresses that gap by fine-tuning PaddleOCR-VL on a purpose-built dataset of synthetic and real Aksara Jawa images.

## Team

| Name | Role |
|---|---|
| Yudha Setyaji | ML Engineer & Project Lead — architecture, training pipeline, scripts, repository |
| Budi Haryono | ML Engineer — AI Studio training, data collection, annotation |

## Model and Dataset

| Resource | Link |
|---|---|
| Model | [setilanaji/PaddleOCR-VL-Aksara-Jawa](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa) |
| Dataset | [setilanaji/aksara-jawa-ocr](https://huggingface.co/datasets/setilanaji/aksara-jawa-ocr) |

Base model: `PaddlePaddle/PaddleOCR-VL` (v1.0, Apache-2.0)  
Fine-tuning method: LoRA via PaddleFormers and ERNIEKit  
Training platform: Baidu AI Studio (V100 16GB)

## Use Cases

**Cultural heritage digitization.** The National Library of Indonesia (PNRI) holds over 12,000 digitised Javanese manuscripts on its Khastara platform. Automated OCR would dramatically reduce the cost of transcription and indexing for philological and historical research.

**Education technology.** Aksara Jawa is a compulsory subject in the SD through SMA curriculum across Yogyakarta Special Region, Central Java, and East Java. OCR capability enables interactive learning tools, automated exercise marking, and digital dictionary lookup from images.

**Government signage accessibility.** Yogyakarta Special Region has a regional regulation requiring Aksara Jawa on all official government signage alongside Latin script. Screen readers and accessibility applications require OCR to interpret these signs for visually impaired users.

**Cultural tourism.** Visitors to Kraton Yogyakarta, Kraton Surakarta, Prambanan, and heritage museums encounter Aksara Jawa inscriptions and labels. A mobile application capable of reading and translating these inscriptions would significantly improve visitor engagement.

**Historical document digitization.** Javanese-language newspapers such as Bromartani (1855–1932) were printed in Aksara Jawa and remain unindexed. OCR enables full-text search and digital preservation at scale.

## Dataset

| Split | Images | Layout | Augmentation | Seed |
|---|---|---|---|---|
| train | 1,000 | 591 single-line, 409 multi-line | mixed | 42 |
| eval | 150 | single-line only | light | 42 |

The training set consists of synthetic images generated from the Noto Sans Javanese font (SIL OFL 1.1) using a custom pipeline that renders text from a 303-line Javanese Unicode corpus. Images span eight background tones, six ink colours, four font sizes, and three augmentation severity levels to simulate real-world capture conditions including smartphone photography, aged paper, and low-quality scans.

The evaluation set uses light augmentation only to enable accurate scoring. Both sets are fully reproducible from the source corpus and generation script using a fixed random seed.

Real manuscript images from PNRI Khastara are used for supplementary training but are not redistributed. See `docs/data_report.md` for a complete description of data sources, annotation methodology, and quality control procedures.

### Annotation Format

PaddleOCR detection format (`Label.txt`):

```
aksara_0001.jpg	[{"transcription": "ꦲꦤꦏ꧀ꦏꦶ", "points": [[45,12],[207,12],[207,45],[45,45]], "label": "aksara_jawa", "illegibility": false}]
```

VLM training format (`ground_truth.jsonl`):

```json
{
  "image": "aksara_0001.jpg",
  "conversations": [
    {"role": "user", "content": "Baca teks Aksara Jawa dalam gambar ini."},
    {"role": "assistant", "content": "ꦲꦤꦏ꧀ꦏꦶ"}
  ]
}
```

### Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="setilanaji/aksara-jawa-ocr",
    repo_type="dataset",
    local_dir="data/"
)
```

## Repository Structure

```
paddleocr-aksara-jawa/
├── assets/
│   ├── NotoSansJavanese-Regular.ttf   # SIL OFL 1.1 — not committed, download via setup
│   └── corpus_jawa.txt                # 303-line Javanese Unicode corpus
├── annotation/                        # PPOCRLabel annotation workspace
├── docs/
│   └── data_report.md                # Training data construction report
├── scripts/
│   ├── generate_aksara.py             # Synthetic image generator
│   └── evaluate.py                    # CER/WER evaluation (coming soon)
├── training/                          # AI Studio training configuration
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

> Dataset files are hosted on Hugging Face and excluded from this repository. Run the setup instructions below to generate data locally or download from HuggingFace.

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone git@github.com:setilanaji/paddleocr-aksara-jawa.git
cd paddleocr-aksara-jawa
uv sync

# Download font
curl -L -o assets/NotoSansJavanese-Regular.ttf \
  "https://github.com/notofonts/javanese/raw/main/fonts/ttf/NotoSansJavanese-Regular.ttf"
```

## Data Generation

```bash
# Training set
uv run python scripts/generate_aksara.py \
  --count 1000 --output data/synthetic/ --seed 42

# Evaluation set
uv run python scripts/generate_aksara.py \
  --count 150 --output data/eval/ --eval
```

Available flags:

| Flag | Description | Default |
|---|---|---|
| `--count` | Number of images to generate | 1000 |
| `--output` | Output directory | data/synthetic/ |
| `--seed` | Random seed for reproducibility | None |
| `--mode` | Layout: `single`, `multiline`, `mixed` | mixed |
| `--augmentation` | Severity: `light`, `medium`, `heavy`, `mixed` | mixed |
| `--eval` | Eval preset: seed=42, light augmentation, single-line | false |

## Fine-tuning

Training uses PaddleFormers on Baidu AI Studio. PaddlePaddle 3.2.1 or above is required.

> Use `PaddlePaddle/PaddleOCR-VL` (v1.0). Fine-tuning v1.5 is currently unsupported — see [issue #17589](https://github.com/PaddlePaddle/PaddleOCR/issues/17589).

```bash
# AI Studio notebook setup
pip install paddlepaddle==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install "paddleocr>=3.0.0"

# LoRA fine-tuning
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  model_name_or_path=PaddlePaddle/PaddleOCR-VL \
  train_dataset_path=./data/ground_truth.jsonl \
  lora_rank=16 \
  lora_alpha=32 \
  learning_rate=2e-4 \
  num_train_epochs=5
```

## Evaluation

```bash
uv run python scripts/evaluate.py \
  --model_path ./output/aksara_model \
  --eval_dir ./data/eval \
  --output results.json
```

Reported metrics: Character Error Rate (CER), Word Error Rate (WER), exact match rate.

## Results

Training in progress. Results will be published after May 2026.

| Metric | Baseline | Fine-tuned |
|---|---|---|
| CER | — | TBD |
| WER | — | TBD |
| Exact match | — | TBD |

## Script Complexity

Aksara Jawa is structurally more complex than most scripts targeted by existing OCR systems. Key challenges include:

- **Pasangan** — When a consonant is followed by another consonant without an intervening vowel, the second consonant takes a stacked form beneath the first. There are 20 pasangan forms, each visually distinct from the base character.
- **Sandhangan** — Vowel sounds are represented by diacritical marks that appear above, below, before, or after a base character, sometimes combining multiple marks on a single character.
- **Vertical stacking** — Characters can stack two or three levels deep, requiring recognition of vertical spatial relationships absent from Latin-script OCR tasks.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

The Noto Sans Javanese font is copyright Google Inc., licensed under the SIL Open Font License 1.1.

## Citation

```bibtex
@misc{setilanaji2026aksarajawa,
  title   = {PaddleOCR-VL Fine-tuned for Aksara Jawa OCR},
  author  = {Yudha Setyaji and Budi Haryono},
  year    = {2026},
  url     = {https://github.com/setilanaji/paddleocr-aksara-jawa}
}
```

## Acknowledgements

- PaddleOCR team for the base model and fine-tuning framework
- Google Noto Fonts project for the Javanese typeface
- Perpustakaan Nasional Republik Indonesia for digitised manuscript access via Khastara
- British Library Endangered Archives Programme for Javanese manuscript digitization