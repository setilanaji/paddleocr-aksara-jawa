# RunPod Setup — PaddleOCR-VL Aksara Jawa Fine-tuning

End-to-end guide for launching a RunPod GPU pod, installing PaddleFormers /
ERNIEKit, pulling this repo + data, and running the LoRA fine-tune.

## 0. Prerequisites

- RunPod account with credit loaded (https://runpod.io) — budget ~$10–15 for
  a 5-epoch LoRA run on A100 40GB Community Cloud
- HuggingFace account + token with **write** access
  (https://huggingface.co/settings/tokens) — used for pushing the finished
  model checkpoint back
- SSH key registered with RunPod (Settings → SSH Public Keys)

## 1. Launch the pod

Go to **https://runpod.io/console/deploy** and pick:

| Field | Recommended | Notes |
|---|---|---|
| **Cloud type** | Community Cloud | ~40% cheaper than Secure Cloud, fine for a training job |
| **GPU** | 1× A100 SXM 40GB | ~$0.80–1.20/hr. RTX 4090 24GB works (~$0.40/hr) but needs batch size changes — see §6b |
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | ships with PyTorch + CUDA already |
| **Container disk** | 40 GB | headroom for model weights + conda env |
| **Volume disk** | 20 GB (optional, persistent across restarts) | stores checkpoints if you stop/resume |
| **Volume mount path** | `/workspace` | default |
| **Environment variables** | `HF_TOKEN=<your_hf_token>` | optional, convenient for `huggingface-cli login` |
| **Expose TCP ports** | 22 | for SSH |
| **Expose HTTP ports** | 8888 | for Jupyter (optional) |

Click **Deploy On-Demand** → wait ~30 s for the pod to come up.

## 2. Connect

Two options in the RunPod pod page → **Connect** button:

**Option A — SSH** (recommended):
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```
RunPod gives you the exact command.

**Option B — Jupyter Lab** (click **Connect to Jupyter Lab** in the UI). Terminal tab has the same shell.

## 3. Install PaddlePaddle + ERNIEKit

Inside the pod:

```bash
cd /workspace

# PaddlePaddle GPU build matching CUDA 12.x
pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# PaddleFormers (ships ERNIEKit CLI) + PaddleOCR
pip install paddleformers>=0.3.0
pip install "paddleocr>=3.0.0"

# Sanity check — should print a Paddle version and no CUDA errors
python -c "import paddle; paddle.utils.run_check()"
which paddleformers-cli
```

If `paddle.utils.run_check()` reports CUDA detected, you're good.

## 4. Get the repo

```bash
cd /workspace
git clone https://github.com/setilanaji/paddleocr-aksara-jawa.git
cd paddleocr-aksara-jawa

# Install uv (faster than pip for project deps)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Bring in project Python deps (Pillow, requests, jiwer, pymupdf, ...)
uv sync
```

## 5. Get the data

Three options from cheapest to most traffic:

### 5a. Regenerate synthetic inside the pod (recommended)

Since the synthetic pipeline is fully reproducible from `seed=42`, the 2,000
pure-synthetic images don't need to travel over the network. Only the 134
real-manuscript backgrounds (`data/real/`) do.

```bash
# 1. Download the font (already in the repo via assets/fonts/, but grab it
#    if you cloned a shallow copy that excluded large files)
ls assets/fonts/   # should show NotoSansJavanese-{Regular,Bold}.ttf

# 2. Re-harvest the 134 real manuscripts from Dreamsea (~35 min, idempotent)
uv run python scripts/dreamsea_collect.py \
  --script Javanese \
  --country Indonesia \
  --limit 30 \
  --output annotation/dreamsea_manifests.txt

uv run python scripts/collect_manuscripts.py \
  --manifests annotation/dreamsea_manifests.txt \
  --output data/real/ \
  --max_pages_per_manifest 5 \
  --width 1200

# 3. Regenerate pure-synthetic (2000) + semi-synthetic (1000) + eval (150)
uv run python scripts/generate_aksara.py \
  --count 2000 --output data/synthetic/ --seed 42 \
  --pasangan_ratio 0.2 --mode mixed --erniekit \
  --image_dir ./data/synthetic

uv run python scripts/generate_aksara.py \
  --count 1000 --output data/semi_synthetic/ --seed 142 \
  --pasangan_ratio 0.2 --mode mixed --erniekit \
  --real_bg_dir data/real/ --real_bg_ratio 1.0 \
  --image_dir ./data/semi_synthetic

uv run python scripts/generate_aksara.py \
  --count 150 --output data/eval/ --eval --erniekit \
  --image_dir ./data/eval

# 4. Merge into ERNIEKit training files
uv run python scripts/convert_format.py \
  --input data/synthetic/ground_truth.jsonl data/semi_synthetic/ground_truth.jsonl \
  --image_dir data/synthetic/ data/semi_synthetic/ \
  --output training/ocr_vl_sft-train_aksara_jawa.jsonl --shuffle

uv run python scripts/convert_format.py \
  --input data/eval/ground_truth.jsonl \
  --image_dir data/eval/ \
  --output training/ocr_vl_sft-test_aksara_jawa.jsonl
```

### 5b. rsync from local (fastest if your internet is fine)

From your laptop:

```bash
rsync -avP -e "ssh -p <port>" \
  data/ root@<pod-ip>:/workspace/paddleocr-aksara-jawa/data/
rsync -avP -e "ssh -p <port>" \
  training/ root@<pod-ip>:/workspace/paddleocr-aksara-jawa/training/
```

Roughly 500 MB total. At 50 Mbps upload: ~90 seconds.

### 5c. Push dataset to HuggingFace Hub, pull from pod

Only worth it if you'll run training on multiple pods or want the dataset
publicly citable:

```bash
# On your laptop
huggingface-cli login
huggingface-cli upload setilanaji/aksara-jawa-ocr data/ \
  --repo-type=dataset --commit-message="2000+1000+134+150 v1"

# In the pod
huggingface-cli login     # paste HF_TOKEN
huggingface-cli download setilanaji/aksara-jawa-ocr \
  --repo-type=dataset --local-dir=data/
```

## 6. Run the fine-tune

### 6a. A100 40GB — config runs as-is

```bash
cd /workspace/paddleocr-aksara-jawa

# Launch (will download PaddlePaddle/PaddleOCR-VL base weights on first run)
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train training/aksara_jawa_lora_config.yaml

# Monitoring in another terminal / Jupyter cell
watch -n 5 nvidia-smi
tail -f PaddleOCR-VL-Aksara-Jawa-lora/visualdl_logs/*.log
```

Expected wall-clock: ~2–3 hours for 5 epochs on A100 40GB at batch size 8
with gradient accumulation 8 (effective batch 64).

### 6b. RTX 4090 24GB — smaller batch, more accumulation

Edit `training/aksara_jawa_lora_config.yaml`:

```yaml
per_device_train_batch_size: 2        # was 8
per_device_eval_batch_size: 2         # was 8
gradient_accumulation_steps: 32       # was 8 — keeps effective batch size at 64
```

Wall-clock ~4–6 hours. Doable but A100 is better ROI per dollar if you value
your time.

## 7. Push the finished checkpoint to HuggingFace

```bash
# In the pod, after training finishes
huggingface-cli login    # if not done already

# Upload the LoRA weights + config to your HF model repo
huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa \
  PaddleOCR-VL-Aksara-Jawa-lora/ \
  --commit-message="LoRA run 1: rank=16 lr=2e-4 epochs=5 seed=42"
```

## 8. Stop the pod

**Important — RunPod bills you for the pod even when idle.**

- **Stop** (cheap) if you used a Volume Disk: pod is paused, volume persists,
  you pay ~$0.01/hr for storage. Resume later without re-downloading anything.
- **Terminate** (free thereafter): everything except the mounted volume is
  deleted. Cheaper if you've already pushed the checkpoint to HF.

```bash
# On the pod, after verifying the HF upload completed
runpodctl stop pod $RUNPOD_POD_ID
# OR click "Stop" in the RunPod dashboard
```

## 9. Verify + evaluate

Back on your laptop (or still in the pod):

```bash
uv sync --extra eval    # pulls torch + transformers

uv run python scripts/evaluate.py \
  --model_path setilanaji/PaddleOCR-VL-Aksara-Jawa \
  --baseline_model_path PaddlePaddle/PaddleOCR-VL \
  --eval_dir data/eval/ \
  --output results.json
```

The diff table `Δ CER` shows the fine-tune's improvement over the base model —
paste the number into `README.md` Results and `docs/data_report.md`.

## 10. Quick reference — one-screen recipe

```bash
# in the pod
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleformers "paddleocr>=3.0.0"
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone https://github.com/setilanaji/paddleocr-aksara-jawa.git && cd paddleocr-aksara-jawa
uv sync
# ... data transfer via 5a/b/c ...
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train training/aksara_jawa_lora_config.yaml
huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa PaddleOCR-VL-Aksara-Jawa-lora/
```
