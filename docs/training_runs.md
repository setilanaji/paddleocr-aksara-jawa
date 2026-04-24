# Training Runs

Versioned log of fine-tuning runs for PaddleOCR-VL on Aksara Jawa. Each entry pins the codebase commit, dataset composition, hyperparameters, and what we learned for the next iteration.

---

## v1 — 2026-04-25 (initial release)

**HF model**: [setilanaji/PaddleOCR-VL-Aksara-Jawa](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa)
**Repo commit**: tag pending (`v1.0-lora`)
**Hardware**: 1× A100 SXM 80GB (RunPod Community Cloud, ~$1.50 total)
**Wall-clock**: 23 min 42 s (training only; including pod setup + data pull + export, ~45 min)

### Base model

`PaddlePaddle/PaddleOCR-VL` v1.0 (Apache-2.0). Per [hackathon issue #17858](https://github.com/PaddlePaddle/PaddleOCR/issues/17858), v1.5 is also accepted but `modeling_paddleocr_vl.py` for v1 is what was already calibrated and tested in this project.

### Dataset

| Tier | Count | Source |
|---|---|---|
| Synthetic | 2,000 | `scripts/generate_aksara.py` — Faker corpus + 4 Aksara Jawa fonts at multiple sizes/spacings |
| Semi-synthetic | 1,000 | Same generator with `--real_bg_dir data/real/` — synthetic glyphs composited onto real manuscript backgrounds |
| **Train total** | **3,000** | shuffled, seed=42 |
| Eval (held out) | 150 | Synthetic with separate seed; same pipeline, no real-bg compositing |

The 120 real-manuscript candidates from `scripts/pick_eval_candidates.py` are excluded from semi-synthetic backgrounds (`--exclude data/eval_real_candidates.txt`) so they remain a clean held-out set for v2 once Label Studio annotation is complete.

### Hyperparameters

| Param | Value | Diff vs upstream | Why |
|---|---|---|---|
| Method | LoRA | — | Resource-constrained run; full-param needs 80 GB+ |
| LoRA rank | 16 | upstream 8 | Higher capacity for a novel script the base hasn't seen |
| LoRA alpha | 32 | not set upstream | 2× rank, standard |
| Learning rate | 2e-4 | upstream 5e-4 | More stable on a small (3k) dataset |
| Schedule | cosine, 1% warmup | same | — |
| Epochs | 5 | upstream 2 | Small dataset benefits from more passes |
| Per-device batch | 8 | same | — |
| Grad accumulation | 8 | same | — |
| **Effective batch** | **64** | same | — |
| Seed | 42 | — | reproducibility |
| `do_eval` | **false** | upstream true | paddleformers VL-LoRA eval path raises `NameError: GPTModel` at step 200; standalone eval planned post-training |

### Trained parameters

```
Trainable parameters : 1.50e+07  (1.63%)
Frozen parameters    : 9.06e+08
Total parameters     : 9.21e+08
```

### Training metrics

- **Final loss (averaged)**: 0.1887
- **Steps**: 235 total (47 per epoch × 5 epochs)
- **Throughput**: ~10.6 samples/sec, ~6 sec/step on A100
- **Loss trajectory**: 1.93 (step 1) → 1.32 (step 5) → 0.99 (step 10) → 0.05–0.13 oscillation past step 80
- **VRAM peak**: 16 GB / 80 GB → **very underutilized**; v2 should use larger batch

### Quantitative evaluation

**Pending.** Two compounding upstream issues blocked CER/WER computation in this session:

1. `paddleformers-cli export` produces **HF safetensors** in `<output_dir>/export/`; `paddleocr.PaddleOCRVL` (paddle-native inference) requires **paddle inference format** (`.pdmodel` + `.pdiparams`).
2. Loading the HF safetensors via `transformers.AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` hits three layered breaks because the upstream `modeling_paddleocr_vl.py` predates transformers 5.x: `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS`, `AttributeError: 'RotaryEmbedding' object has no attribute 'compute_default_rope_parameters'`, and so on. Patching reveals the next break each time.

Decision: ship v1 without numerical eval, fix the export path in v2 (either find a paddleformers→paddle-inference converter or write the safetensors→paddle conversion). See [`runpod_tutorial.md` §7](runpod_tutorial.md#7-evaluate-against-the-synthetic-eval-set) Troubleshooting for the full error chain.

### Qualitative observations

The base `PaddlePaddle/PaddleOCR-VL` produces empty / non-Aksara-Jawa output on synthetic eval images (sample command in `runpod_tutorial.md`). This establishes the baseline: any non-trivial Aksara Jawa output from v1 is improvement signal — the magnitude awaits the eval fix.

### Lessons learned (carry into v2)

- **GPU under-utilized.** A100 80GB held 16 GB peak. v2 should bump `per_device_train_batch_size: 8 → 32`, drop `gradient_accumulation_steps: 8 → 2` (keeps effective batch at 64), and raise `pre_alloc_memory: 16 → 48`. Expected ~3× wall-clock improvement.
- **Wall-clock estimate was wrong.** The tutorial inherited "2–3 hours" from the upstream Bengali config tuned for a 16k-sample dataset; with 3k samples at the same epochs/batch, it's ~23 min. Tutorial corrected.
- **Mid-training eval is broken** in paddleformers for VL-LoRA. `do_eval: false` + `evaluation_strategy: "no"` is the only safe configuration today. `save_strategy: steps` keeps periodic checkpoints regardless.
- **The export → inference path needs work.** Pick one of: (a) find/file an upstream paddleformers fix to emit paddle inference format directly; (b) write a one-shot safetensors→pdmodel converter; (c) accept transformers-only inference and pin compatible versions of every dep. Recommendation for v2: (a) first, fall back to (b).
- **Label Studio R2-backed annotation flow** (already documented in `annotation_guide.md`) is the v2 unlock. ~50 real annotations is the threshold the project README cites for meaningful real-data eval.

### Next iterations

- **v2 — real annotations**: complete Label Studio annotation of the 120 candidate manuscript pages, regenerate corpus with `scripts/extract_real_corpus.py`, retrain with combined synth + real, **and solve the eval path** so we can report CER.
- **v3 — scale + tuning**: with the v2 eval pipeline working, sweep LoRA rank (16, 32, 64) and learning rate (1e-4, 2e-4, 5e-4); 30-min runs make this cheap.

---

<!-- Template for future entries

## vX — YYYY-MM-DD (one-line summary)

**HF model**: …
**Repo commit**: …
**Hardware**: …
**Wall-clock**: …

### Dataset
### Hyperparameters
### Trained parameters
### Training metrics
### Quantitative evaluation
### Qualitative observations
### Lessons learned
### Next iterations
-->
