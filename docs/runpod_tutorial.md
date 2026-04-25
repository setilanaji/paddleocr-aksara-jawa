# RunPod Training Tutorial — Fine-tuning PaddleOCR-VL for Aksara Jawa

Linear walkthrough for someone running the fine-tune for the first time. If
you've done this before, the terser reference is [`runpod_execute.md`](runpod_execute.md).

**What you get after ~3 hours and ~$1.50:** a LoRA-fine-tuned `PaddleOCR-VL`
checkpoint that recognises Aksara Jawa, pushed to Hugging Face, with CER
numbers on the synthetic eval set.

---

## 0. Before you start

Do these on your **laptop**, not on RunPod.

### 0.1. You need accounts / tokens for

| Service | Why | Link |
|---|---|---|
| [RunPod](https://runpod.io) | Runs the GPU | add ~$10 credit |
| [Hugging Face](https://huggingface.co/settings/tokens) | Hosts the fine-tuned model | create a token with **write** access |
| Cloudflare R2 | Stores the training bundle | already configured — your `annotation/.env` holds the keys |

### 0.2. Sanity-check locally

```bash
# In the repo root on your laptop:
git status                   # should be clean
git log --oneline -1         # most recent commit should match remote
git push origin main         # the pod clones from GitHub, so push first
echo $HF_TOKEN | head -c 8   # should print the first 8 chars of your HF token

# Confirm the training bundle exists on R2 — pod will pull from here
rclone ls r2:aksara-jawa-images/training-bundle/ --max-depth 1
# You should see:
#   data/
#   training/
```

### 0.3. Grab two secrets you'll paste into the pod later

You'll paste the following values into the pod's shell in a moment — **do
not** put them in git, screenshots, or Slack:

- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID` (from `annotation/.env`)
- `HF_TOKEN` (already set in your local shell)

Keep a terminal open with `cat annotation/.env` ready to copy from.

---

## 1. Launch the RunPod pod

Use the field values from [`runpod_setup.md §1`](runpod_setup.md). The short
version:

1. Go to `https://runpod.io/console/deploy`
2. **Cloud type:** Community Cloud
3. **GPU:** 1× A100 SXM 40GB (≈$0.80–1.20/hr)
4. **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
5. **Container disk:** 40 GB
6. **Volume disk:** 20 GB mounted at `/workspace` (persists if you stop the pod)
7. **Environment variables:** add `HF_TOKEN=<your_hf_token>`
8. **Expose TCP port:** 22 (for SSH)

Click **Deploy**. Wait ~1 minute for "Running" state.

---

## 2. SSH into the pod

RunPod dashboard → your pod → **Connect** → copy the SSH command. Looks like (use your ssh key location):

```bash
ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_ed25519
```

Paste into your laptop terminal. You should land at a `root@...:/workspace#`
prompt.

---

## 3. Bootstrap the environment (~3 min, one-time per pod)

```bash
cd /workspace
git clone https://github.com/setilanaji/paddleocr-aksara-jawa.git
cd paddleocr-aksara-jawa
bash scripts/runpod_bootstrap.sh
```

**What this does:**
- Installs PaddlePaddle 3.3.1 (the version `paddleformers` needs)
- Installs `paddleformers` (which ships the `paddleformers-cli train` binary)
- Installs `uv` + syncs the repo's virtualenv
- Idempotent — re-run it if anything goes wrong

**Expected end:** you'll see `bootstrap complete` and a `python -c "import paddle"` check passing. If you see errors about missing CUDA libraries, the pod template was wrong — stop the pod and launch a fresh one with the template listed in §1.

---

## 4. Pull the training data from R2 (~30 s)

The data is pre-built on your laptop and mirrored to R2. The pod just pulls it.

Open a named tmux session so the pull doesn't die if your SSH disconnects:

```bash
apt-get update && apt-get install -y tmux           
tmux new -s data
```

Inside that tmux, paste the following **after replacing the `...` with your R2 values**. Each `export` is on its own line on purpose — a single `export A=x B=${A}` expands `${A}` *before* it's assigned and silently leaves `B=` empty, which shows up later as `https://.r2.cloudflarestorage.com: no such host`.

```bash
cd /workspace/paddleocr-aksara-jawa
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=aksara-jawa-images
export R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Sanity check — must show your account ID, not an empty host.
echo "endpoint is: $R2_ENDPOINT"

# provider=Other, not Cloudflare — pod rclone is often old enough that
# `provider=Cloudflare` fails with `s3 provider "Cloudflare" not known`.
# Upgrade path if you want the nicer value: curl https://rclone.org/install.sh | bash
rclone config create r2 s3 provider=Other env_auth=false \
  access_key_id="$R2_ACCESS_KEY_ID" secret_access_key="$R2_SECRET_ACCESS_KEY" \
  region=auto endpoint="$R2_ENDPOINT" >/dev/null

rclone copy "r2:$R2_BUCKET/training-bundle/" /workspace/paddleocr-aksara-jawa/ \
  --transfers 16 --progress --s3-no-check-bucket && \
echo "=== DATA DONE ==="
```

**Expected output:** a progress bar showing ~50 MB transferred, finishing with `=== DATA DONE ===`.

If it looks done but hangs on the last file (e.g. `3161 / 3162, 100%` with a file stuck at 0 B/s for more than a minute), that's a transient Cloudflare hiccup on one object. `Ctrl+C` and re-run the same `rclone copy` — it's idempotent, skips the files already present, and finishes the last one in seconds. If it hangs again on the same file, retry with tighter timeouts:

```bash
rclone copy "r2:$R2_BUCKET/training-bundle/" /workspace/paddleocr-aksara-jawa/ \
  --transfers 4 --contimeout 30s --timeout 60s --progress --s3-no-check-bucket && \
echo "=== DATA DONE ==="
```

Sanity-check what landed before moving on:

```bash
wc -l training/ocr_vl_sft-train_aksara_jawa.jsonl   # expect 3000
ls data/synthetic/ | wc -l                          # expect ~2000
ls data/eval/ | wc -l                               # expect ~150
```

Detach from tmux: `tmux detach` or `Ctrl+b` then `d`. You can reattach anytime with `tmux attach -t data`.

---

## 5. Start training (~25 min on A100 with 3k-sample dataset)

> **Wall-clock note.** The upstream reference config was tuned for a **16k-sample** Bengali set, where 5 epochs takes ~2 hr. With this project's ~3k-sample dataset, the same config finishes in ~25 min on A100. Expect proportional growth as you add real annotations: doubling the dataset doubles the runtime at the same epoch count.

### 5a. One-time pod prep (~30 s)

Two fixes the R2 bundle alone doesn't cover:

1. **Image path resolution.** paddleformers resolves `images:` relative to the jsonl's directory (`training/`), but image paths inside are like `data/synthetic/aksara_0001.jpg`. Without this, every record errors with `is not a valid url or file path` and every example is skipped. Symlink `training/data → ../data` so the paths resolve:

   ```bash
   cd /workspace/paddleocr-aksara-jawa
   ln -sfn /workspace/paddleocr-aksara-jawa/data training/data
   ls -la training/data/synthetic/aksara_0001.jpg   # must resolve to a real file
   ```

2. **JSONL schema.** If the bundle on R2 was pushed before commit `7a8d997` ("emit paddleformers messages+images schema"), the jsonl uses the old `image_info`/`text_info` keys and training fails with `preprocess data error: 'messages'` on every record. Check and regenerate if needed:

   ```bash
   head -1 training/ocr_vl_sft-train_aksara_jawa.jsonl | \
     python3 -c "import json,sys; print(list(json.loads(sys.stdin.read()).keys()))"
   # Expected: ['messages', 'images']
   # If you see ['image_info', 'text_info'], regenerate from source:
   uv run python scripts/convert_format.py \
       --input data/synthetic/ground_truth.jsonl data/semi_synthetic/ground_truth.jsonl \
       --image_dir data/synthetic/ data/semi_synthetic/ \
       --output training/ocr_vl_sft-train_aksara_jawa.jsonl --shuffle
   uv run python scripts/convert_format.py \
       --input data/eval/ground_truth.jsonl \
       --image_dir data/eval/ \
       --output training/ocr_vl_sft-test_aksara_jawa.jsonl
   ```

   Re-run the `head -1 | python3 -c ...` check — now it must print `['messages', 'images']`.

   After confirming, re-push the fixed bundle to R2 so the next pod doesn't repeat this.

### 5b. Launch training

New tmux session:

```bash
tmux new -s train
```

Inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train \
    training/aksara_jawa_lora_config.yaml 2>&1 | tee /workspace/train.log
```

**Watch the first ~30 s:** there should be no `preprocess data error: 'messages'` warnings and no `is not a valid url or file path` errors. If you see either, stop (`Ctrl+C`) and redo 5a — launching a full run with a broken dataset burns A100 hours on zero signal.

**What happens:**

- First ~2 minutes: downloads base `PaddlePaddle/PaddleOCR-VL` weights (~2 GB from HF)
- After that: training logs print every step. You'll see `loss`, `learning_rate`, and progress through 5 epochs
- Loss for LoRA on PaddleOCR-VL starts **already low** (~0.1–0.3, not the 4+ you'd see in LLM full-param training) because the base VL model already handles images well — only the ~1.6% LoRA params adapt to Aksara Jawa. Expect final loss in the **0.02–0.08** range. If you see loss values > 1 at epoch 0.3+, something is wrong with the data pipeline (re-check §5a). If you see `ppl` (perplexity) stuck at ~1.0 with loss near 0, that can mean labels are all masked — verify at §7 eval that the model actually emits Aksara Jawa.

Detach with `Ctrl+b d`. Monitor from any shell without re-entering tmux:

```bash
tail -f /workspace/train.log
```

Optional — watch GPU utilisation in a separate tmux:

```bash
tmux new -s gpu
watch -n 5 nvidia-smi
# Expect: >90% GPU utilisation, ~30 GB VRAM used on A100 40GB
```

---

## 5c. Merge the LoRA adapter into a standalone model (~3 min)

`paddleformers-cli train` writes only the **adapter** (`peft_model-*.safetensors` + `lora_config.json`). Loading that directly with `transformers.AutoModelForCausalLM.from_pretrained` fails — transformers expects a complete model with custom `.py` modeling files. The upstream solution is `paddleformers-cli export`, which merges adapter + base into a complete checkpoint.

```bash
cd /workspace/paddleocr-aksara-jawa
paddleformers-cli export training/aksara_jawa_lora_export.yaml
```

**What happens:**
- Loads base `PaddlePaddle/PaddleOCR-VL` (~2 GB, cached from training)
- Applies the LoRA adapter on top
- Writes the merged complete model to `./PaddleOCR-VL-Aksara-Jawa-lora/export/`

**Verify the merged weights are present:**

```bash
ls PaddleOCR-VL-Aksara-Jawa-lora/export/ | grep -E "model.*safetensors|config\.json"
```

**Expect:** `config.json`, `model-*.safetensors` (one or more shards), `model.safetensors.index.json`.

### Add the custom .py modeling files (paddleformers doesn't copy them at export time)

`paddleformers-cli export` merges the weights but does **not** copy the four `*_paddleocr_vl.py` files even if `copy_custom_file_list` is set in the export config. Pull them straight from the base model HF repo into the export dir:

```bash
huggingface-cli download PaddlePaddle/PaddleOCR-VL configuration_paddleocr_vl.py modeling_paddleocr_vl.py image_processing_paddleocr_vl.py processing_paddleocr_vl.py inference.yml --local-dir PaddleOCR-VL-Aksara-Jawa-lora/export/
```

> Keep this command on **one line**. Multi-line backslash continuation is fragile here — a single trailing space after `\` silently turns the rest of the lines into separate failed commands.

Verify all five landed:

```bash
ls PaddleOCR-VL-Aksara-Jawa-lora/export/ | grep -E "\.py$|inference\.yml"
```

**Expect:** all five filenames listed. **This is now the dir you upload to HF, not the parent.**

---

## 6. Upload the merged model to Hugging Face

Once you've run §5c and `export/` exists, push that subdir:

```bash
cd /workspace/paddleocr-aksara-jawa && \
huggingface-cli login --token "$HF_TOKEN" && \
huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa \
    PaddleOCR-VL-Aksara-Jawa-lora/export/ \
    --commit-message="LoRA v1 merged: rank=16 alpha=32 lr=2e-4 epochs=5 seed=42"
```

**Expected output:** per-file progress bars (the model shards are the bulk, ~2 GB total), ending with a commit SHA + a link to `https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa/tree/main`. Open it to confirm the `.py` files landed alongside the safetensors.

---

## 7. Evaluate against the synthetic eval set

Two backends are wired up. **Try Path A first** — it matches the official ERNIE SFT docs and uses the same paddleocr stack the model was trained for.

### 7a. Path A — paddleocr-native (recommended)

The official PaddleOCR-VL SFT guide loads a fine-tuned safetensors checkpoint with `--vl_rec_model_dir` pointing at the merged LoRA export dir. `scripts/predict_paddleocr.py` wraps that and writes one prediction per eval image to a file; `evaluate.py --predictions` then scores it.

Run in the **system Python** (paddleocr lives there, not in the uv venv):

```bash
cd /workspace/paddleocr-aksara-jawa && \
python3 scripts/predict_paddleocr.py \
    --model_dir /workspace/paddleocr-aksara-jawa/PaddleOCR-VL-Aksara-Jawa-lora/export \
    --eval_dir data/eval/ \
    --output predictions_finetuned.txt && \
python3 scripts/predict_paddleocr.py \
    --model_dir "$(huggingface-cli download PaddlePaddle/PaddleOCR-VL --quiet)" \
    --eval_dir data/eval/ \
    --output predictions_baseline.txt && \
uv run python scripts/evaluate.py \
    --predictions predictions_finetuned.txt \
    --eval_dir data/eval/ \
    --output results_finetuned.json && \
uv run python scripts/evaluate.py \
    --predictions predictions_baseline.txt \
    --eval_dir data/eval/ \
    --output results_baseline.json
```

The paddleocr download for the baseline lands the base model into `~/.cache/huggingface/...`; we hand that path back to `predict_paddleocr.py` so the loader uses the same code path for both.

If paddleocr complains about the checkpoint format, you've hit the safetensors-vs-paddle-inference issue noted in v1's `training_runs.md`. Fall through to Path B.

### 7b. Path B — transformers fallback

`scripts/evaluate.py` loads the model via `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`. The eval extra now pins `transformers>=4.55,<5.0` because PaddleOCR-VL's modeling code predates transformers 5.x (the previous `>=5.6` pin was a misdiagnosis — see Troubleshooting).

```bash
cd /workspace/paddleocr-aksara-jawa && \
uv sync --extra eval && \
uv run --extra eval python scripts/evaluate.py \
    --model_path setilanaji/PaddleOCR-VL-Aksara-Jawa \
    --baseline_model_path PaddlePaddle/PaddleOCR-VL \
    --eval_dir data/eval/ \
    --output results.json
```

**What to look for in either path:**

- **Baseline CER** on synthetic eval: high (PaddleOCR-VL has not seen Aksara Jawa)
- **Fine-tuned CER** on synthetic eval: near 0 (<5% CER, usually <1%) — eval is in-distribution
- The delta is the v1 improvement signal worth quoting in the model card

**Synthetic CER is not the competition number.** The hackathon requires a **real-data** eval set. Build it by annotating the 120 candidates in Label Studio (see [`annotation_guide.md`](annotation_guide.md)). Once you have ≥50 real annotations, rerun whichever path worked above with `--eval_dir data/real/`.

---

## 8. Stop the pod

RunPod dashboard → your pod → **Stop**.

- **Running:** ~$0.80–1.20/hr
- **Stopped:** ~$0.01/hr (just `/workspace` volume storage)
- **To resume:** hit Start. `/workspace` keeps everything — no re-bootstrap needed.

---

## Troubleshooting

### "AccessDenied: CreateBucket" on rclone
Your R2 token is object-scoped (good). Make sure every rclone command in §4
includes `--s3-no-check-bucket`. It tells rclone to skip the bucket-creation
preflight that your token can't perform.

### `s3 provider "Cloudflare" not known`
Pod rclone is older than the release that added the `Cloudflare` provider
string. Use `provider=Other` (as §4 does now), or upgrade rclone with
`curl https://rclone.org/install.sh | bash`.

### `dial tcp: lookup .r2.cloudflarestorage.com: no such host`
The endpoint URL has no account ID — note the leading dot. This happens when
you chain all `export`s on one command (`export A=x B=${A}`); `${A}` is
expanded before it's assigned, so `R2_ENDPOINT` ends up as
`https://.r2.cloudflarestorage.com`. Split each `export` onto its own line
and re-run. `echo "endpoint is: $R2_ENDPOINT"` should show the account ID.

### `tmux: command not found`
Not every RunPod template has tmux preinstalled. The first block in §4
installs it with `apt-get`. If you skipped that, run it now.

### rclone copy hangs at 100% on the last file
Byte total reads `50.379 MiB / 50.379 MiB` and file count is `3161 / 3162`,
but one small object sits at 0 B/s for more than a minute. Transient R2
hiccup on a single key. `Ctrl+C` and re-run — `rclone copy` skips what's
already on disk. If it hangs on the same file twice in a row, add
`--transfers 4 --contimeout 30s --timeout 60s` so the client gives up and
retries that key instead of waiting forever.

If `Ctrl+C` has no effect (web terminals sometimes swallow it) and `kill -9 <pid>`
doesn't either, check process state: `ps -p <pid> -o stat,cmd`. `Z`/`Zl`
(zombie) means the process is already dead — drop the session with
`tmux kill-session -t data` and the kernel reaps it. `D`/`Dl` (uninterruptible
sleep) means it's blocked in a syscall the kernel can't interrupt — only the
pod restart will clear it.

### Training: `Error: ... is not a valid url or file path`
Every record is skipped because paddleformers resolves image paths relative
to the jsonl's directory (`training/`), but the paths inside are
`data/synthetic/...`. Symlink so they resolve:
```bash
ln -sfn /workspace/paddleocr-aksara-jawa/data training/data
```

### Training: `preprocess data error: 'messages'`
Dataset is in the old `image_info`/`text_info` schema; the current config
wants the `messages`/`images` schema (commit `7a8d997`). Regenerate on the
pod from `data/*/ground_truth.jsonl` — see §5a step 2.

### Evaluation: `does not appear to have a file named configuration_paddleocr_vl.py`
You uploaded the raw LoRA adapter directory instead of the merged model.
`paddleformers-cli train` writes only `peft_model-*.safetensors` + `lora_config.json`,
which `transformers.AutoModelForCausalLM.from_pretrained` cannot load on its own.
Run `paddleformers-cli export` first (see §5c) and upload `<output_dir>/export/`,
not `<output_dir>/`.

### Evaluation (Path A): `No Paddle model files were found in '<...>/export'`
PaddleX defaulted to the classic Paddle inference engine and didn't find `.pdmodel`/`.pdiparams` because we exported HF safetensors. The fix is to tell PaddleOCRVL to use the **GenAI backend** (which reads safetensors). The supported kwargs vary by paddleocr version, so first introspect, then pass whichever backend kwarg exists. Drop into the pod and copy:

```bash
python3 -c "import paddleocr; print('paddleocr', paddleocr.__version__)" && \
python3 << 'EOF'
import inspect
from paddleocr import PaddleOCRVL
sig = inspect.signature(PaddleOCRVL.__init__)
print("PaddleOCRVL kwargs:")
for name, p in sig.parameters.items():
    default = p.default if p.default is not inspect.Parameter.empty else "<required>"
    print(f"  {name} = {default}")
EOF
```

Look for a kwarg matching `*backend*`, `*genai*`, or `*server_url*`. The AMD ROCm article (Jan 2026) shows the YAML form:
```yaml
VLRecognition:
  model_dir: ./checkpoint-5000
  genai_config:
    backend: native
```
The Python equivalent on current paddleocr is typically `vl_rec_backend="native"` or `vl_rec_genai_config={"backend": "native"}`. Once you know the right kwarg, add it to `scripts/predict_paddleocr.py` in the `PaddleOCRVL(...)` call and re-run.

If no GenAI kwarg exists in your installed paddleocr, upgrade: `pip install -U "paddleocr[doc-parser]" "paddlex[ocr]"`. Or fall back to Path B (`scripts/evaluate.py --model_path ...`).

### Evaluation: any of `KeyError: 'default'`, `compute_default_rope_parameters` missing, or `create_causal_mask() got 'inputs_embeds'`
All three are symptoms of running PaddleOCR-VL on a transformers version newer than the model's modeling code expects. The model was authored against transformers 4.x; chasing the breaks "upward" into 5.x leads through one new error after another (we tried up to 5.6 in v1 and gave up).

The fix is to pin **down**, not up. The eval extra now uses `transformers>=4.55,<5.0` (per the official inference snippet in [HF discussion #1](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/1)). If you've manually upgraded, revert:
```bash
uv sync --extra eval        # picks up the pinned <5.0 transformers
```
If a future release of `modeling_paddleocr_vl.py` ships transformers-5 compatibility, lift the upper bound.

### Evaluation: `requires the protobuf library but it was not found`
The Llama tokenizer in `transformers` decodes `tokenizer.model` (sentencepiece) via `protobuf` and doesn't pull it as a hard dep. Either `uv pip install protobuf` for an immediate fix, or add `"protobuf>=4.0"` to the `eval` extra in `pyproject.toml` so `uv sync --extra eval` doesn't strip it.

### Evaluation: `cannot import name 'cached_assets_path' from 'huggingface_hub'`
`paddleocr` still calls `cached_assets_path`, which `huggingface_hub` removed
in 1.0. The project now pins `huggingface_hub>=0.20,<1.0` at top-level
(safe because `transformers<5.0` in the eval extra is happy with hub 0.x too).
Run `uv sync && uv pip list | grep huggingface` to confirm a 0.x version is
installed. If uv still resolves >=1.0, clear the lock and re-sync:
`rm uv.lock && uv sync --extra eval`. Lift the pin once paddleocr ships a
hub-1.x-compatible release (`pip index versions paddleocr`).

### Training crashes at first eval step with `NameError: name 'GPTModel' is not defined`
Upstream bug in `paddleformers/trainer/trainer.py:4460` — the VL-LoRA branch
of `evaluation_loop` references `GPTModel` without importing it, so the very
first periodic eval (step 200 by default) crashes the run. Loss before the
crash is real, the LoRA adapters are valid up to the last `save_steps`
checkpoint. Fix in `training/aksara_jawa_lora_config.yaml`:
```yaml
do_eval: false
evaluation_strategy: "no"
```
`save_strategy: steps` still snapshots the model — you'll have a checkpoint
to resume from. Real evaluation runs once at §7 via `scripts/evaluate.py`,
which doesn't go through the broken trainer path.

### "CUDA out of memory"
You landed on a smaller GPU than A100 40GB. Either:
- Stop the pod, launch a new one with A100 40GB, or
- Edit `training/aksara_jawa_lora_config.yaml` — reduce `per_device_train_batch_size` by 2× and double `gradient_accumulation_steps` to keep the effective batch the same.

### SSH disconnects while training
tmux detached — the training keeps running. SSH back in and reattach:
```bash
tmux attach -t train
```

### "HF 401 Unauthorized" on model upload
Your `HF_TOKEN` env var isn't set in the pod (or doesn't have **write** scope). Fix:
```bash
export HF_TOKEN=<your_new_token>
huggingface-cli login --token $HF_TOKEN
```

### "No CUDA device found" after bootstrap
Wrong pod template. Stop the pod, launch a new one with the exact template name from §1.

### Training loss doesn't decrease
Either:
- The training data didn't land — check `ls /workspace/paddleocr-aksara-jawa/training/ocr_vl_sft-train_aksara_jawa.jsonl` exists and has 3,000 lines (`wc -l`)
- Config is pointing at the wrong paths — diff `training/aksara_jawa_lora_config.yaml` against `training/paddleocr-vl_lora_16k_config.yaml` (the upstream reference) and look at `train_dataset.data_path`

---

## What "success" looks like

- HF model repo `setilanaji/PaddleOCR-VL-Aksara-Jawa` has a checkpoint committed
- `train.log` shows final loss ≤ 0.5
- `results.json` shows synthetic-eval CER ≤ 5%, clear ΔCER vs the base model
- The Gradio demo at `demo/app.py` loads the new model without errors

Once you have real annotations, rerun step 7 pointing at `data/real/` — that real-world CER is the number that matters for the competition.

---

## Next, once this all works

Iterate:
1. Annotate more manuscripts in Label Studio
2. Run `scripts/extract_real_corpus.py --write` to pull real transcriptions into a corpus file
3. Regenerate synthetic + semi-synthetic with the richer corpus
4. Push new bundle to R2 (`rclone copy ... --s3-no-check-bucket`)
5. Restart the pod, pull new data, train v2
6. Compare v1 vs v2 CER on the real eval set — that ΔCER is worth points on the "experiment thoroughness" rubric dimension
