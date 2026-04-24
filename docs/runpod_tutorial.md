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

RunPod dashboard → your pod → **Connect** → copy the SSH command. Looks like:

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
tmux new -s data
```

Inside that tmux, paste the following **after replacing the `...` with your R2 values**:

```bash
cd /workspace/paddleocr-aksara-jawa && \
export R2_ACCOUNT_ID=... \
       R2_ACCESS_KEY_ID=... \
       R2_SECRET_ACCESS_KEY=... \
       R2_BUCKET=aksara-jawa-images \
       R2_ENDPOINT=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com && \
rclone config create r2 s3 provider=Cloudflare env_auth=false \
  access_key_id="$R2_ACCESS_KEY_ID" secret_access_key="$R2_SECRET_ACCESS_KEY" \
  region=auto endpoint="$R2_ENDPOINT" >/dev/null && \
rclone copy "r2:$R2_BUCKET/training-bundle/" /workspace/paddleocr-aksara-jawa/ \
  --transfers 16 --progress --s3-no-check-bucket && \
echo "=== DATA DONE ==="
```

**Expected output:** a progress bar showing ~50 MB transferred, finishing with `=== DATA DONE ===`.

Detach from tmux: `Ctrl+b` then `d`. You can reattach anytime with `tmux attach -t data`.

---

## 5. Start training (~2–3 hours)

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

**What happens:**

- First ~2 minutes: downloads base `PaddlePaddle/PaddleOCR-VL` weights (~2 GB from HF)
- After that: training logs print every step. You'll see `loss`, `learning_rate`, and progress through 5 epochs
- Loss should decrease steadily — expect ~4.5 at epoch 0, ~0.3 at epoch 5 for a healthy run

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

## 6. Upload the fine-tuned model to Hugging Face

Once training finishes (`tmux attach -t train` — you'll see `=== TRAINING DONE ===` or similar), push the checkpoint:

```bash
cd /workspace/paddleocr-aksara-jawa && \
huggingface-cli login --token $HF_TOKEN && \
huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa \
    PaddleOCR-VL-Aksara-Jawa-lora/ \
    --commit-message="LoRA v1: rank=16 alpha=32 lr=2e-4 epochs=5 seed=42"
```

**Expected output:** a commit SHA and a link to `https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa/tree/main`. Open the link in a browser to confirm the files landed.

---

## 7. Evaluate against the synthetic eval set

```bash
cd /workspace/paddleocr-aksara-jawa && \
uv run --extra eval python scripts/evaluate.py \
    --model_path setilanaji/PaddleOCR-VL-Aksara-Jawa \
    --baseline_model_path PaddlePaddle/PaddleOCR-VL \
    --eval_dir data/eval/ \
    --output results.json
```

**What to look for:**

- **Baseline CER** (base model before fine-tuning) on synthetic eval: usually very high or essentially random — PaddleOCR-VL hasn't seen Aksara Jawa
- **Fine-tuned CER** on synthetic eval: should be near 0 (<5% CER, usually <1%) — the model has been trained on synthetic, so eval is in-distribution

**This synthetic CER is not your competition number.** The hackathon requires a **real-data** evaluation set. You build that by annotating the 120 candidates in Label Studio (see [`annotation_guide.md`](annotation_guide.md)). Once you have ≥50 real annotations, rerun `scripts/evaluate.py` with `--eval_dir data/real/` instead.

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
