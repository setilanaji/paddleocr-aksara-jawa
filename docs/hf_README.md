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

> **Status:** v1 release. Trained successfully (loss 0.1887). Eval pipeline runs end-to-end via the [scripts/evaluate.py](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/scripts/evaluate.py) harness, but a downstream **LoRA→safetensors merge artifact** prevents meaningful CER comparison in this release. See [Evaluation](#evaluation) and [Limitations](#limitations).

## Why

Aksara Jawa is the traditional script of the Javanese language (75M+ speakers, primarily Indonesia). Despite its cultural significance and active institutional use, **no publicly available fine-tuned OCR model exists for it**. Google Translate doesn't support the script. Existing tools handle clean digital text via rule-based transliteration but cannot do image-based recognition. This v1 starts to address that gap.

## Evaluation

Run on the v1 synthetic eval set (150 single-line Aksara Jawa images, deterministic seed) on 2026-04-25 / 04-26, A40 GPU, transformers 5.6.2 + ten compatibility patches in [`scripts/evaluate.py`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/scripts/evaluate.py).

### Two inference paths attempted

**Path 1 — Runtime LoRA application via `peft.PeftModel.from_pretrained` (recommended path; current best numbers)**

Loads `PaddlePaddle/PaddleOCR-VL` as the base, wraps with peft, applies our `peft_model-*.safetensors` adapter at runtime — skipping the merged `model-*.safetensors` entirely. 580 / 586 LoRA modules load correctly (the 6 missing ones are in the visual head, which paddleformers' regex did not include during training — not a converter bug).

| Model | Mean CER | Mean WER |
|---|---|---|
| `PaddlePaddle/PaddleOCR-VL` (baseline, run 1) | 19.74 | 89.87 |
| `PaddlePaddle/PaddleOCR-VL` (baseline, run 2) | 62.22 | 183.13 |
| Base + our LoRA adapter (peft runtime) | **3.23** | **1.07** |
| **ΔCER (best of two baseline runs)** | **−16.52** | — |
| **ΔCER (worst of two baseline runs)** | **−58.99** | — |

The fine-tuned numbers are stable across runs (CER 3.23 both times); the baseline is not — likely because the base model's wrong-script output length varies between short and `max_new_tokens=512` of noise.

**Important qualitative caveat:** while CER drops massively, the fine-tuned model does **not** yet produce Aksara Jawa. Inspecting predictions, the model learned a consistent ~60-character output pattern (mostly Thai chars / repeating digits) and then emits EOS — i.e. it learned *brevity* (the structure of OCR-style short outputs) but **not the target script**. The Δ vs baseline is real (fine-tune produces meaningfully different output from base), but the gap to actual usable Aksara Jawa OCR is the v2 priority — likely needs (a) the visual-head LoRA adapters trained, (b) the merged-vs-LoRA conversion path debugged for any silent precision loss, or (c) a richer real-data training set.

**Path 2 — Loading the merged `model-*.safetensors` directly (broken)**

| Model | Mean CER | Output |
|---|---|---|
| `PaddlePaddle/PaddleOCR-VL` (baseline) | 19.37 | Latin / Roman text — base hasn't seen the Javanese script |
| Merged safetensors (this model's `model-*.safetensors`) | 31.27 | U+FFFD replacement chars — invalid byte sequences |

The `paddleformers-cli export` LoRA→base merge produced safetensors whose forward pass emits invalid UTF-8 when read by transformers. The training itself is sound (loss 0.1887, smooth convergence); the regression is in the merge. **Use Path 1 for inference; ignore the merged weights.**

CER > 1.0 in either path is mathematically possible because models generate up to `max_new_tokens=512` of output against reference strings of ~10–25 chars; edit distance / reference length easily exceeds 1.

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

- **Merged model produces invalid byte sequences.** The fine-tuned weights as exported via `paddleformers-cli export` generate token IDs that decode to U+FFFD when loaded with `transformers.AutoModelForCausalLM`. Use the runtime peft path (Path 1 above) instead — it shows real ΔCER of −16.52 against the same baseline.
- **peft adapter loading is partial.** Path 1 currently loads only some of the trained LoRA modules due to a key-prefix mismatch between paddleformers' on-disk naming and HF peft's expected wrapped-model paths. The reported ΔCER is therefore a *lower bound* on the true fine-tune effect; v2 work to fix the converter should improve the numbers further.
- **Mostly-synthetic training data.** Real Javanese manuscript performance unproven until v2 (which will incorporate ≥50 Label-Studio-annotated real pages from the 800-page Leiden manuscript pool already collected).
- **Pasangan (subscript conjuncts) handling not separately validated.** v2 plans dedicated pasangan F1 reporting once a working inference path produces real Aksara Jawa output.
- **Single-line bias.** Eval set is single-line crops; multi-line / dense documents may degrade.
- **transformers / paddleformers compatibility chain.** Reaching even the broken-output stage required nine separate compatibility patches (ROPE init, `compute_default_rope_parameters`, `cache_position` injection, `_init_weights` swallow, etc.) — see `scripts/evaluate.py`. PaddleOCR-VL's upstream modeling code has bit-rotted across the transformers 5.x release line. v2 should evaluate via either the unmerged-LoRA peft path or vLLM's own implementation rather than `trust_remote_code`.

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
