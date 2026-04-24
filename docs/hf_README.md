---
language:
- jv
- id
license: apache-2.0
base_model: PaddlePaddle/PaddleOCR-VL
library_name: paddleformers
tags:
- ocr
- aksara-jawa
- javanese
- paddleocr-vl
- lora
- fine-tuned
- low-resource
pipeline_tag: image-to-text
---

<!--
This is the model card content for the Hugging Face repo
`setilanaji/PaddleOCR-VL-Aksara-Jawa`. To publish it:

  huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa \
      docs/hf_README.md README.md \
      --commit-message="model card v1"

Don't confuse this with the repo-level README.md at the root of this codebase.
-->


# PaddleOCR-VL fine-tuned for Aksara Jawa (Javanese script) — v1

LoRA fine-tune of [`PaddlePaddle/PaddleOCR-VL`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) for Aksara Jawa (Javanese traditional script). Submitted to the [PaddleOCR Global Derivative Model Challenge — Hackathon 10th](https://github.com/PaddlePaddle/PaddleOCR/issues/17858).

> **Status:** v1 release, trained but quantitative eval pending. See [Limitations](#limitations).

## Why

Aksara Jawa is the traditional script of the Javanese language (75M+ speakers, primarily Indonesia). Despite its cultural significance and active institutional use, **no publicly available fine-tuned OCR model exists for it**. Google Translate doesn't support the script. Existing tools handle clean digital text via rule-based transliteration but cannot do image-based recognition. This v1 starts to address that gap.

## Quick comparison (qualitative)

| Image | Base PaddleOCR-VL | This model (v1) |
|---|---|---|
| Synthetic single-line Aksara Jawa | empty / Latin garbage | _quantitative metrics pending paddle-format export — see Limitations_ |

A full quantitative CER/WER table will land with v2 (see [training_runs.md](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/docs/training_runs.md)).

## Training data

| Tier | Count | Source |
|---|---|---|
| Synthetic | 2,000 | Generated via [`scripts/generate_aksara.py`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/scripts/generate_aksara.py): Faker corpus + 4 Aksara Jawa fonts, varied size/spacing |
| Semi-synthetic | 1,000 | Same generator with `--real_bg_dir`: synthetic glyphs composited onto real manuscript backgrounds |
| **Train total** | **3,000** | Shuffled, seed=42 |
| Eval (held out) | 150 | Synthetic with separate seed |

## Training procedure

| Param | Value |
|---|---|
| Method | LoRA (PEFT) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable params | 15.0 M (1.63% of 921 M) |
| Learning rate | 2e-4, cosine schedule, 1% warmup |
| Epochs | 5 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 8 |
| Effective batch | 64 |
| Seed | 42 |
| Precision | bf16 |
| Framework | [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers) (paddleformers-cli train) |
| Hardware | 1× A100 SXM 80GB (RunPod) |
| Wall-clock | 23 min 42 s (235 steps) |
| Final loss | 0.1887 (averaged over run) |

Full config: [`training/aksara_jawa_lora_config.yaml`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/training/aksara_jawa_lora_config.yaml).

## Intended use

- Recognising Aksara Jawa text from images (single-line crops to short paragraphs)
- Hackathon prototype, research, and education on low-resource scripts

**Not intended for** safety-critical or production OCR. Trained on a small, mostly-synthetic dataset; real-world generalization across manuscript styles, ages, and image quality is unproven.

## Limitations

- **Quantitative CER/WER not yet reported.** `paddleformers-cli export` writes HuggingFace safetensors; `paddleocr.PaddleOCRVL` (the paddle-native runtime) requires paddle inference format (`.pdmodel`/`.pdiparams`). Loading the safetensors via `transformers.AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` hits multiple incompatibilities between the upstream `modeling_paddleocr_vl.py` and transformers ≥5.x. Resolving the export path is the v2 priority. See [`docs/training_runs.md`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/docs/training_runs.md#quantitative-evaluation) for the full chain of issues.
- **Mostly-synthetic training data.** Real Javanese manuscript performance may differ significantly until v2 (which will incorporate ≥50 Label-Studio-annotated real pages).
- **Pasangan (subscript conjuncts) handling not separately validated.** v2 plans dedicated pasangan F1 reporting.
- **Single-line bias.** Eval set is single-line crops; multi-line / dense documents may degrade.

## Versions

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-04-25 | Initial release. 3 k-sample LoRA fine-tune. No quantitative eval yet. |

## Citation

```bibtex
@misc{setyaji_paddleocr_aksara_jawa_2026,
  author       = {Yudha Setyaji},
  title        = {PaddleOCR-VL Fine-tuned for Aksara Jawa OCR},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa}},
  note         = {Submitted to PaddleOCR Global Derivative Model Challenge — Hackathon 10th}
}
```

## License

Apache 2.0 — matching the base model.

## Acknowledgements

- The PaddleOCR team for the base model and the PaddleFormers fine-tuning framework
- Hackathon organisers for the structured opportunity to release this work openly
- The maintainers of the Javanese-script fonts used in synthetic data generation
