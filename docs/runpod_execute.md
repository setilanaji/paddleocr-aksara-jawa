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

## Step 1 — Data pipeline (tmux `data`, ~45 min)

```bash
tmux new -s data
```

Paste inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
uv run python scripts/dreamsea_collect.py \
    --script Javanese --country Indonesia --limit 30 \
    --output annotation/dreamsea_manifests.txt && \
uv run python scripts/collect_manuscripts.py \
    --manifests annotation/dreamsea_manifests.txt \
    --output data/real/ --max_pages_per_manifest 5 --width 1200 && \
uv run python scripts/generate_aksara.py \
    --count 2000 --output data/synthetic/ --seed 42 \
    --pasangan_ratio 0.2 --mode mixed --erniekit \
    --image_dir ./data/synthetic && \
uv run python scripts/generate_aksara.py \
    --count 1000 --output data/semi_synthetic/ --seed 142 \
    --pasangan_ratio 0.2 --mode mixed --erniekit \
    --real_bg_dir data/real/ --real_bg_ratio 1.0 \
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

If `data/synthetic/` already has 2000 images from a previous run, drop the
`generate_aksara.py --count 2000 ...` line to save ~3 min.

**Detach:** `Ctrl+b` then `d`.
**Reattach:** `tmux attach -t data`.

Proceed to step 2 after `=== DATA DONE ===` prints.

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

Uploads **only** `data/synthetic/` (pure-synthetic, fully open — Noto Sans
Javanese SIL OFL 1.1). Explicitly excludes `data/real/` and
`data/semi_synthetic/` because the Dreamsea source terms restrict
redistribution (see `docs/data_report.md` §2.2).

```bash
tmux new -s upload
```

Paste inside:

```bash
cd /workspace/paddleocr-aksara-jawa && \
huggingface-cli login --token $HF_TOKEN && \
huggingface-cli upload setilanaji/aksara-jawa-ocr data/synthetic/ \
    --repo-type=dataset \
    --commit-message="v2 pure-synthetic: 2000 images, multi-font (Noto Regular+Bold), 20% pasangan-stress"
```

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
