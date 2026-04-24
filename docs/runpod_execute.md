# RunPod Training Playbook — execute in this order

The **one-time** RunPod pod launch and dependency install is covered in
[`runpod_setup.md`](runpod_setup.md). This doc is the **runtime** playbook:
data → train → upload, run from inside the already-bootstrapped pod.

## Prerequisites (check before starting)

```bash
cd /workspace/paddleocr-aksara-jawa
which tmux paddleformers-cli uv        # all three must print paths
python -c "import paddle; print(paddle.__version__)"   # expect 3.3.1
ls data/synthetic/*.jpg | wc -l         # expect 2000 if re-using; 0 is fine
echo $HF_TOKEN | head -c 10             # sanity check HF_TOKEN is set
```

If anything is missing, run `bash scripts/runpod_bootstrap.sh` first.

---

## Step 1 — Pull pre-built training bundle from R2 (~30 seconds)

The data is **built once locally** (seed-pinned, ~700 MB) and mirrored to
Cloudflare R2 at `aksara-jawa-images/training-bundle/`. The pod just pulls it.
This replaces the old 45-minute on-pod regeneration and avoids repeating an
expensive pipeline the local machine has already run.

### 1a. One-time local push (run on your laptop, not the pod)

```bash
# From the repo root on your laptop
set -a; source annotation/.env; set +a

# If rclone isn't configured yet, the /annotation-r2-sync skill set it up;
# otherwise:
rclone config create r2 s3 provider=Cloudflare env_auth=false \
  access_key_id="$R2_ACCESS_KEY_ID" secret_access_key="$R2_SECRET_ACCESS_KEY" \
  region=auto endpoint="$R2_ENDPOINT" >/dev/null

# Push training-ready data. Mirrors repo layout so a single rclone copy on the
# pod lands files in the exact paths the training config expects.
rclone copy data/synthetic      "r2:$R2_BUCKET/training-bundle/data/synthetic"      --transfers 8 --s3-no-check-bucket
rclone copy data/semi_synthetic "r2:$R2_BUCKET/training-bundle/data/semi_synthetic" --transfers 8 --s3-no-check-bucket
rclone copy data/eval           "r2:$R2_BUCKET/training-bundle/data/eval"           --transfers 8 --s3-no-check-bucket
rclone copyto --s3-no-check-bucket training/ocr_vl_sft-train_aksara_jawa.jsonl "r2:$R2_BUCKET/training-bundle/training/ocr_vl_sft-train_aksara_jawa.jsonl"
rclone copyto --s3-no-check-bucket training/ocr_vl_sft-test_aksara_jawa.jsonl  "r2:$R2_BUCKET/training-bundle/training/ocr_vl_sft-test_aksara_jawa.jsonl"
```

Re-run only the tiers that changed — `rclone copy` skips unchanged files.

### 1b. Pod-side pull (tmux `data`, ~30 seconds)

Copy `annotation/.env` to the pod (`/workspace/paddleocr-aksara-jawa/annotation/.env`) or paste the `R2_*` values into the pod's shell, then:

```bash
tmux new -s data
```

Paste inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
set -a; source annotation/.env; set +a && \
rclone config create r2 s3 provider=Other env_auth=false \
  access_key_id="$R2_ACCESS_KEY_ID" secret_access_key="$R2_SECRET_ACCESS_KEY" \
  region=auto endpoint="$R2_ENDPOINT" >/dev/null && \
rclone copy "r2:$R2_BUCKET/training-bundle/" /workspace/paddleocr-aksara-jawa/ \
  --transfers 16 --progress --s3-no-check-bucket && \
echo "=== DATA DONE ==="
```

> `provider=Other` — pod rclone is often old enough that `Cloudflare` isn't a recognised provider. `Other` works on every version. Laptop-side (§1a) keeps `Cloudflare` since local rclone is typically current.

**Detach:** `Ctrl+b` then `d`.
**Reattach:** `tmux attach -t data`.

Proceed to step 2 after `=== DATA DONE ===` prints.

### 1c. Fallback: regenerate on the pod (only if R2 is unavailable)

If R2 access is blocked or `.env` isn't on the pod, regenerate from scratch.
This takes ~5 minutes (synthetic is fast; real-image download is skipped —
use pre-built `data/real/` or rclone it separately). Commands match the
current local pipeline including the eval-candidate exclusion:

```bash
cd /workspace/paddleocr-aksara-jawa && \
uv run python scripts/generate_aksara.py \
    --count 2000 --output data/synthetic/ --seed 42 \
    --pasangan_ratio 0.2 --mode mixed --erniekit \
    --image_dir ./data/synthetic && \
uv run python scripts/generate_aksara.py \
    --count 1000 --output data/semi_synthetic/ --seed 142 \
    --pasangan_ratio 0.2 --mode mixed --erniekit \
    --real_bg_dir data/real/ --real_bg_ratio 1.0 \
    --exclude data/eval_real_candidates.txt \
    --image_dir ./data/semi_synthetic && \
uv run python scripts/generate_aksara.py \
    --count 150 --output data/eval/ --eval --erniekit \
    --image_dir ./data/eval && \
uv run python scripts/convert_format.py \
    --input data/synthetic/ground_truth.jsonl data/semi_synthetic/ground_truth.jsonl \
    --image_dir data/synthetic/ data/semi_synthetic/ \
    --output training/ocr_vl_sft-train_aksara_jawa.jsonl --shuffle && \
uv run python scripts/convert_format.py \
    --input data/eval/ground_truth.jsonl \
    --image_dir data/eval/ \
    --output training/ocr_vl_sft-test_aksara_jawa.jsonl && \
echo "=== DATA DONE ==="
```

The semi-synthetic regeneration requires `data/real/` (800 Leiden pages) to be
present on the pod — either rclone them from `r2:$R2_BUCKET/training-bundle/data/real/`
first, or do the full primary-path pull in §1b.

---

## Step 2 — Training (tmux `train`, ~2–3 hr)

```bash
tmux new -s train
```

Paste inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train \
    training/aksara_jawa_lora_config.yaml 2>&1 | tee /workspace/train.log
```

First run downloads the base `PaddlePaddle/PaddleOCR-VL` weights (~2 GB from
HuggingFace) before the first step. Detach with `Ctrl+b d`. Monitor progress
with `tail -f /workspace/train.log` from any shell.

Parallel GPU monitoring (optional):

```bash
tmux new -s gpu
watch -n 5 nvidia-smi
```

---

## Step 3 — Public dataset upload (tmux `upload`, parallel with training)

All current tiers are redistributable:

- `data/synthetic/` + `data/semi_synthetic/` + `data/eval/` — rendered from Noto Sans Javanese ([SIL OFL 1.1](https://openfontlicense.org/), redistribution permitted)
- `data/real/` — 800 Leiden pages under [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) (replaced the previous Dreamsea pool; see `docs/data_report.md` §2.3 / §6.4 for the migration note)
- `training/*.jsonl` — our own derived artifacts

The upload is driven by `scripts/hf_push_dataset.py`, which mirrors the layout in `docs/data_report.md` §6, attaches per-subset license info in the dataset card, and skips `eval/real_v2/` when annotations don't yet exist.

```bash
tmux new -s upload
```

Paste inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
huggingface-cli login --token $HF_TOKEN && \
uv run python scripts/hf_push_dataset.py --dry-run && \
uv run python scripts/hf_push_dataset.py
```

The dry-run prints the file counts per tier and the destination paths — use it to sanity-check before a multi-hundred-megabyte upload. To iterate just one tier (e.g. after regenerating semi-synthetic):

```bash
uv run python scripts/hf_push_dataset.py --only semi_synthetic
uv run python scripts/hf_push_dataset.py --only readme   # re-render the dataset card without touching data
```

After annotations are exported to `data/real/ground_truth.jsonl`, rerun the full push — `eval/real_v2/` will be included automatically.

---

## Step 4 — Model upload (after training finishes)

```bash
cd /workspace/paddleocr-aksara-jawa && \
huggingface-cli upload setilanaji/PaddleOCR-VL-Aksara-Jawa \
    PaddleOCR-VL-Aksara-Jawa-lora/ \
    --commit-message="LoRA run 1: rank=16 alpha=32 lr=2e-4 epochs=5 seed=42"
```

---

## Step 5 — Evaluate fine-tuned model

```bash
cd /workspace/paddleocr-aksara-jawa && \
uv run --extra eval python scripts/evaluate.py \
    --model_path setilanaji/PaddleOCR-VL-Aksara-Jawa \
    --baseline_model_path PaddlePaddle/PaddleOCR-VL \
    --eval_dir data/eval/ \
    --output results.json
```

This prints per-tier CER/WER + pasangan F1 + delta vs the base model. Paste
the summary into `README.md` Results table and `docs/data_report.md`.

---

## Step 6 — Stop the pod

After verifying HF uploads completed (`huggingface-cli` prints a commit SHA),
stop the pod from the RunPod web dashboard. You're billed ~$0.01/hr for
volume storage while stopped; $0.44/hr while running.

To resume later without losing anything: the `/workspace` volume persists.

---

## tmux cheat sheet

```
tmux ls                  # list sessions
tmux attach -t <name>    # reattach
Ctrl+b d                 # detach
Ctrl+b [  then q         # scroll mode + quit
exit                     # kill current session
```
