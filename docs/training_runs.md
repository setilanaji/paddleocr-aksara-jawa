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

Run on **2026-04-25** (5 months after training) on a fresh A40 RunPod instance, transformers 5.6.2 + nine custom compatibility patches now committed in [`scripts/evaluate.py`](../scripts/evaluate.py).

**Results on the v1 synthetic eval set (n=150, deterministic seed):**

| Model | Mean CER | Mean WER | Output character |
|---|---|---|---|
| `PaddlePaddle/PaddleOCR-VL` (baseline) | **19.37** | **14.45** | Latin / Roman text — base model emits valid Unicode but cannot read Aksara Jawa |
| `setilanaji/PaddleOCR-VL-Aksara-Jawa` (v1) | **31.27** | **1.00** | U+FFFD replacement chars only — invalid byte sequences |

CER > 1.0 because both models generate up to `max_new_tokens=512` of wrong output against ~10–25-char references.

**The fine-tuned model produces invalid bytes, not Aksara Jawa.** Training itself was sound (loss 0.1887, smooth convergence, real LoRA adapter weights in `peft_model-*.safetensors`). The regression is in the **`paddleformers-cli export` LoRA→base merge step**: the merged `model-*.safetensors` it writes produce token IDs that decode to U+FFFD when read by transformers + the upstream `modeling_paddleocr_vl.py`. The same code path on the **base** model produces valid (if wrong-script) output, ruling out the inference harness as the cause.

#### What it took to reach this conclusion

The transformers + `trust_remote_code` path required all of the following to even reach `model.generate()`:

1. Pin `transformers>=5.6` (model code uses `inputs_embeds`, the post-5.6 kwarg)
2. Drop top-level `huggingface_hub<1.0` pin (transformers 5.6 needs ≥1.5; uv venv is isolated from system paddleocr's hub)
3. Pin `torch>=2.4,<2.6` (RunPod's runpod-torch-v240 ships system CUDA 12.4; torch ≥2.6 needs cu126's `cudaGetDriverEntryPointByVersion`)
4. Re-register `ROPE_INIT_FUNCTIONS["default"]` (removed in transformers 5.0; still referenced by model's `RotaryEmbedding.__init__`)
5. Property-alias `RotaryEmbedding.compute_default_rope_parameters → rope_init_fn` (transformers 5.x's `_init_weights` looks for the new name)
6. Wrap `PreTrainedModel._init_weights` in try/except to swallow the rope `AttributeError` after weight load (weights are already loaded; the post-load re-init is defensive only)
7. Inject `cache_position` in `prepare_inputs_for_generation` (transformers 5.6 stopped auto-creating it for trust_remote_code models)
8. Use multimodal-content list message format (plain-string content makes the chat template emit zero image placeholder tokens → "tokens: 0, features N" error)
9. `pip install paddlex[ocr]` for the layout-detection sub-dependencies the VL pipeline pulls in

Commits implementing the chain: [`ad19852`](https://github.com/setilanaji/paddleocr-aksara-jawa/commit/ad19852), [`39113db`](https://github.com/setilanaji/paddleocr-aksara-jawa/commit/39113db), [`469c084`](https://github.com/setilanaji/paddleocr-aksara-jawa/commit/469c084), [`9dfb3ee`](https://github.com/setilanaji/paddleocr-aksara-jawa/commit/9dfb3ee), [`fb741c9`](https://github.com/setilanaji/paddleocr-aksara-jawa/commit/fb741c9). After all of these, the model loads, weights load (620/620), inference runs over all 150 eval images without crashing — but emits invalid UTF-8.

Path A (paddleocr-native) is also exhausted: paddleocr 3.5's `vl_rec_backend` only exposes `native` (which routes to paddle's `paddle_dynamic` engine and demands `.pdmodel`/`.pdiparams` files we don't have) and `*-server` backends. The `transformers` engine that paddlex itself supports for safetensors is not surfaced through the `PaddleOCRVL` Python wrapper.

#### v2 evaluation priorities

1. **Skip the merge entirely** — load `PaddlePaddle/PaddleOCR-VL` as base + `peft.PeftModel.from_pretrained(base, "<repo>", subfolder=...)` to apply our LoRA at runtime. If LoRA application is correct, we'd get real Aksara Jawa output without ever invoking `paddleformers-cli export`.
2. **vLLM with the safetensors** — use vLLM's own PaddleOCR-VL implementation (per AMD's Jan 2026 article); bypasses both `trust_remote_code` and the merge.
3. **Re-export with a different toolchain** — try `peft.PeftModel.merge_and_unload()` then `save_pretrained()` instead of paddleformers-cli; verify byte equality to the safetensors before further work.

### Qualitative observations

The base `PaddlePaddle/PaddleOCR-VL` produces empty / non-Aksara-Jawa output on synthetic eval images (sample command in `runpod_tutorial.md`). This establishes the baseline: any non-trivial Aksara Jawa output from v1 is improvement signal — the magnitude awaits the eval fix.

### Lessons learned (carry into v2)

- **GPU under-utilized.** A100 80GB held 16 GB peak. v2 should bump `per_device_train_batch_size: 8 → 32`, drop `gradient_accumulation_steps: 8 → 2` (keeps effective batch at 64), and raise `pre_alloc_memory: 16 → 48`. Expected ~3× wall-clock improvement.
- **Wall-clock estimate was wrong.** The tutorial inherited "2–3 hours" from the upstream Bengali config tuned for a 16k-sample dataset; with 3k samples at the same epochs/batch, it's ~23 min. Tutorial corrected.
- **Mid-training eval is broken** in paddleformers for VL-LoRA. `do_eval: false` + `evaluation_strategy: "no"` is the only safe configuration today. `save_strategy: steps` keeps periodic checkpoints regardless.
- **The export → inference path needs work.** Pick one of: (a) find/file an upstream paddleformers fix to emit paddle inference format directly; (b) write a one-shot safetensors→pdmodel converter; (c) accept transformers-only inference and pin compatible versions of every dep. Recommendation for v2: (a) first, fall back to (b).
- **Label Studio R2-backed annotation flow** (already documented in `annotation_guide.md`) is the v2 unlock. ~50 real annotations is the threshold the project README cites for meaningful real-data eval.

### Next iterations

- **v2 — fix the merge + add real data**: priority order is (a) reach a working inference path on v1's existing weights via the peft-runtime route above (one pod hour), (b) complete Label Studio annotation of the 120 candidate manuscript pages, (c) regenerate corpus with `scripts/extract_real_corpus.py`, (d) retrain with combined synth + real and report **real** ΔCER vs base.
- **v3 — scale + tuning**: with v2's eval pipeline producing meaningful numbers, sweep LoRA rank (16, 32, 64) and learning rate (1e-4, 2e-4, 5e-4); 30-min runs make this cheap.

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
