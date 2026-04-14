"""
evaluate.py
Evaluation script for PaddleOCR-VL Aksara Jawa fine-tuned model.

Computes Character Error Rate (CER), Word Error Rate (WER), and exact match
accuracy against the ground truth annotations in ground_truth.jsonl.

Usage:
    # Evaluate against ground truth only (no model inference — checks annotation quality)
    uv run python scripts/evaluate.py --gt_only --eval_dir data/eval/

    # Evaluate model predictions against ground truth
    uv run python scripts/evaluate.py \
        --model_path ./output/aksara_model \
        --eval_dir data/eval/ \
        --output results.json

    # Evaluate from a predictions file (one prediction per line, matching GT order)
    uv run python scripts/evaluate.py \
        --predictions predictions.txt \
        --eval_dir data/eval/ \
        --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Metrics ──────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    """Standard dynamic programming edit distance (Levenshtein)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance(hyp, ref) / len(ref)."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return edit_distance(hypothesis, reference) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate = edit_distance(hyp_words, ref_words) / len(ref_words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return edit_distance(hyp_words, ref_words) / len(ref_words)


# ── Ground truth loading ──────────────────────────────────────────────────────

def load_ground_truth(eval_dir: Path) -> list[dict]:
    """
    Load ground truth from ground_truth.jsonl.
    Returns list of {"image": str, "text": str}.
    """
    gt_path = eval_dir / "ground_truth.jsonl"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    records = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Extract assistant response (ground truth text)
        text = ""
        for turn in obj.get("conversations", []):
            if turn.get("role") == "assistant":
                text = turn.get("content", "")
                break
        records.append({
            "image": obj["image"],
            "text":  text,
        })
    return records


# ── Model inference ───────────────────────────────────────────────────────────

def run_model_inference(
    model_path: str,
    eval_dir: Path,
    ground_truth: list[dict],
) -> list[str]:
    """
    Run PaddleOCR-VL inference on eval images.
    Returns list of predicted strings in same order as ground_truth.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: paddleocr not installed. Run: uv sync --extra paddle")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    ocr = PaddleOCR(
        text_detection_model_dir=model_path,
        text_recognition_model_dir=model_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    predictions = []
    for i, record in enumerate(ground_truth):
        img_path = str(eval_dir / record["image"])
        if not os.path.exists(img_path):
            print(f"  WARNING: image not found: {img_path}")
            predictions.append("")
            continue

        try:
            result = ocr.predict(input=img_path)
            # Collect all recognised text lines
            lines = []
            for res in result:
                if hasattr(res, "rec_texts"):
                    lines.extend(res.rec_texts)
                elif isinstance(res, dict) and "rec_texts" in res:
                    lines.extend(res["rec_texts"])
            pred = "\n".join(lines).strip()
        except Exception as e:
            print(f"  WARNING: inference failed for {record['image']}: {e}")
            pred = ""

        predictions.append(pred)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1:>4}/{len(ground_truth)}] {record['image']}")

    return predictions


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    predictions: list[str],
    ground_truth: list[dict],
    verbose: bool = False,
) -> dict:
    """
    Compute CER, WER, exact match, and per-image breakdown.
    """
    assert len(predictions) == len(ground_truth), \
        f"Prediction count ({len(predictions)}) != ground truth count ({len(ground_truth)})"

    total_cer     = 0.0
    total_wer     = 0.0
    exact_matches = 0
    per_image     = []

    for pred, gt in zip(predictions, ground_truth):
        ref  = gt["text"].strip()
        hyp  = pred.strip()

        c = cer(hyp, ref)
        w = wer(hyp, ref)
        exact = (hyp == ref)

        total_cer     += c
        total_wer     += w
        exact_matches += int(exact)

        per_image.append({
            "image":       gt["image"],
            "reference":   ref,
            "hypothesis":  hyp,
            "cer":         round(c, 4),
            "wer":         round(w, 4),
            "exact_match": exact,
        })

        if verbose and not exact:
            print(f"  MISS  {gt['image']}")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")
            print(f"    CER: {c:.4f}  WER: {w:.4f}")

    n = len(ground_truth)
    mean_cer     = total_cer / n
    mean_wer     = total_wer / n
    exact_rate   = exact_matches / n

    return {
        "total_images":   n,
        "mean_cer":       round(mean_cer,   4),
        "mean_wer":       round(mean_wer,   4),
        "exact_matches":  exact_matches,
        "exact_rate":     round(exact_rate, 4),
        "per_image":      per_image,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Aksara Jawa OCR model using CER, WER, and exact match.")
    ap.add_argument("--eval_dir",    type=str, default="data/eval/",
                    help="Directory containing ground_truth.jsonl and images")
    ap.add_argument("--model_path",  type=str, default=None,
                    help="Path to fine-tuned PaddleOCR-VL model directory")
    ap.add_argument("--predictions", type=str, default=None,
                    help="Path to predictions file (one prediction per line)")
    ap.add_argument("--output",      type=str, default="results.json",
                    help="Output path for results JSON")
    ap.add_argument("--gt_only",     action="store_true",
                    help="Check ground truth only (no inference) — for validation")
    ap.add_argument("--verbose",     action="store_true",
                    help="Print mismatches during evaluation")
    ap.add_argument("--top_k",       type=int, default=10,
                    help="Number of worst-performing images to print (default: 10)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"ERROR: eval_dir not found: {eval_dir}")
        sys.exit(1)

    # Load ground truth
    print(f"Loading ground truth from {eval_dir / 'ground_truth.jsonl'}")
    ground_truth = load_ground_truth(eval_dir)
    print(f"  {len(ground_truth)} records loaded")

    # GT-only mode: use GT as both reference and hypothesis (sanity check)
    if args.gt_only:
        print("\nRunning ground truth sanity check (GT vs GT — all metrics should be perfect)...")
        predictions = [r["text"] for r in ground_truth]

    # Predictions from file
    elif args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"ERROR: predictions file not found: {pred_path}")
            sys.exit(1)
        predictions = pred_path.read_text(encoding="utf-8").splitlines()
        if len(predictions) != len(ground_truth):
            print(f"ERROR: predictions ({len(predictions)}) != ground truth ({len(ground_truth)})")
            sys.exit(1)
        print(f"Loaded {len(predictions)} predictions from {pred_path}")

    # Model inference
    elif args.model_path:
        print(f"\nRunning model inference on {len(ground_truth)} images...")
        predictions = run_model_inference(args.model_path, eval_dir, ground_truth)

    else:
        print("ERROR: provide one of --gt_only, --predictions, or --model_path")
        ap.print_help()
        sys.exit(1)

    # Evaluate
    print("\nComputing metrics...")
    results = evaluate(predictions, ground_truth, verbose=args.verbose)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Images evaluated : {results['total_images']}")
    print(f"  Mean CER         : {results['mean_cer']:.4f} ({results['mean_cer']*100:.2f}%)")
    print(f"  Mean WER         : {results['mean_wer']:.4f} ({results['mean_wer']*100:.2f}%)")
    print(f"  Exact matches    : {results['exact_matches']} / {results['total_images']}")
    print(f"  Exact match rate : {results['exact_rate']:.4f} ({results['exact_rate']*100:.2f}%)")
    print("=" * 50)

    # Print worst-performing images
    if not args.gt_only and args.top_k > 0:
        worst = sorted(results["per_image"], key=lambda x: x["cer"], reverse=True)[:args.top_k]
        print(f"\nTop {args.top_k} worst CER images:")
        for item in worst:
            print(f"  {item['image']}  CER={item['cer']:.4f}  WER={item['wer']:.4f}")
            print(f"    REF: {item['reference'][:60]}")
            print(f"    HYP: {item['hypothesis'][:60]}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()