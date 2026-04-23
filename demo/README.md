---
title: PaddleOCR-VL — Aksara Jawa
emoji: ꦲ
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
models:
- setilanaji/PaddleOCR-VL-Aksara-Jawa
- PaddlePaddle/PaddleOCR-VL
tags:
- ocr
- javanese
- aksara-jawa
- paddleocr
- low-resource-language
- cultural-heritage
---

# PaddleOCR-VL — Aksara Jawa OCR Demo

Live demo of the PaddleOCR-VL model fine-tuned for **Aksara Jawa** (Javanese script)
OCR and transcription.

- **Model**: [setilanaji/PaddleOCR-VL-Aksara-Jawa](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa)
- **Code**: [github.com/setilanaji/paddleocr-aksara-jawa](https://github.com/setilanaji/paddleocr-aksara-jawa)
- **Competition**: [PaddleOCR Global Derivative Model Challenge (Hackathon 10th)](https://github.com/PaddlePaddle/PaddleOCR/issues/17858)

## How it works

1. Upload a photo or scan containing Aksara Jawa text (printed, handwritten, or manuscript)
2. The model transcribes it to Unicode Javanese (U+A980–U+A9DF)
3. Copy the Unicode output or use it for downstream processing (TTS, translation, search, etc.)

Base model: `PaddlePaddle/PaddleOCR-VL` (v1.0, Apache-2.0)
Fine-tuning: LoRA via PaddleFormers and ERNIEKit on a mixed synthetic + real Javanese manuscript corpus.

## Deploy your own

Set `MODEL_PATH` in Space Settings to switch between the base model and the fine-tuned Aksara Jawa model:

- `setilanaji/PaddleOCR-VL-Aksara-Jawa` (default, fine-tuned)
- `PaddlePaddle/PaddleOCR-VL` (baseline)

## License

Apache 2.0
