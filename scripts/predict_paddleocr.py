"""
predict_paddleocr.py
Runs the paddleocr-native PaddleOCRVL pipeline over an eval set and writes
one prediction per line, matching the order of ground_truth.jsonl. The output
is consumable by `evaluate.py --predictions ...` to compute CER/WER.

This is the "Path A" eval backend — it uses the same paddleocr/paddlex stack
the model was trained for, sidestepping the transformers compatibility chain.
The official ERNIE SFT docs show that paddleocr loads a fine-tuned safetensors
checkpoint when you pass `vl_rec_model_dir=` pointing at the merged LoRA dir.

Run in the system Python where `paddleocr` is installed (NOT inside `uv run`):

    python3 scripts/predict_paddleocr.py \
        --model_dir ./PaddleOCR-VL-Aksara-Jawa-lora/export \
        --eval_dir data/eval/ \
        --output predictions.txt

Then:

    uv run python scripts/evaluate.py \
        --predictions predictions.txt \
        --eval_dir data/eval/ \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_image_list(eval_dir: Path) -> list[str]:
    gt_path = eval_dir / "ground_truth.jsonl"
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found", file=sys.stderr)
        sys.exit(1)
    images: list[str] = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        images.append(obj["image"])
    return images


def extract_text(result) -> str:
    """
    PaddleOCRVL.predict() returns a generator of result objects whose schema
    varies across pipeline versions. We try the documented `.json` accessor
    and fall back to walking common keys, joining all recognised text spans
    in reading order. Empty string if nothing recognisable.
    """
    try:
        data = result.json
    except AttributeError:
        try:
            data = result.get("res", result)
        except Exception:
            return ""
    if isinstance(data, dict):
        # Common keys produced by VL pipeline outputs
        for key in ("rec_texts", "rec_text", "markdown", "text", "ocr_text"):
            if key in data:
                v = data[key]
                if isinstance(v, list):
                    return "\n".join(str(x) for x in v if x)
                if v:
                    return str(v)
        # Generic fallback: walk for any string-list under "blocks" or similar
        for key in ("blocks", "results", "items"):
            if key in data and isinstance(data[key], list):
                parts = []
                for item in data[key]:
                    if isinstance(item, dict):
                        for k in ("text", "rec_text", "content"):
                            if k in item and item[k]:
                                parts.append(str(item[k]))
                                break
                if parts:
                    return "\n".join(parts)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--model_dir", required=True,
                    help="Path to merged LoRA export dir (contains *.safetensors + inference.yml)")
    ap.add_argument("--eval_dir", default="data/eval/",
                    help="Eval dir containing ground_truth.jsonl and images")
    ap.add_argument("--output", default="predictions.txt",
                    help="Where to write one-prediction-per-line output")
    ap.add_argument("--model_name", default="PaddleOCR-VL-0.9B",
                    help="Model identifier passed to PaddleOCRVL")
    ap.add_argument("--pipeline_version", default="v1",
                    choices=["v1", "v1.5"],
                    help="PaddleOCR-VL pipeline version (v1.5 is current upstream; v1 matches our trained model)")
    ap.add_argument("--backend", default="native",
                    choices=["native", "vllm-server"],
                    help="paddlex inference backend. 'native' reads safetensors directly (required for our HF-format export); 'vllm-server' talks to a separately-launched vLLM server.")
    args = ap.parse_args()

    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        print(
            "ERROR: `paddleocr` not importable. Install in the system env:\n"
            "    pip install 'paddleocr[doc-parser]' 'paddlex[ocr]'\n"
            "and run this script with system `python3`, not `uv run`.",
            file=sys.stderr,
        )
        sys.exit(1)

    eval_dir = Path(args.eval_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    images = load_image_list(eval_dir)
    print(f"Loaded {len(images)} eval images from {eval_dir}")

    print(f"Initialising PaddleOCRVL (model_dir={model_dir}) ...")
    pipe = PaddleOCRVL(
        pipeline_version=args.pipeline_version,
        vl_rec_model_name=args.model_name,
        vl_rec_model_dir=str(model_dir),
        vl_rec_backend=args.backend,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[str] = []

    for i, rel in enumerate(images, 1):
        img_path = eval_dir / rel
        if not img_path.exists():
            print(f"  [{i:>4}/{len(images)}] MISSING {rel}", file=sys.stderr)
            predictions.append("")
            continue
        try:
            results = pipe.predict(str(img_path))
            text = ""
            for r in results:
                text = extract_text(r)
                if text:
                    break
        except Exception as e:
            print(f"  [{i:>4}/{len(images)}] FAIL {rel}: {e}", file=sys.stderr)
            text = ""
        # Sanitize: predictions file is one record per line
        text = text.replace("\n", " ").replace("\r", " ").strip()
        predictions.append(text)
        if i == 1 or i % 25 == 0 or i == len(images):
            print(f"  [{i:>4}/{len(images)}] {rel} -> {text[:60]!r}")

    out_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    print(f"\nWrote {len(predictions)} predictions to {out_path}")
    print(
        "Now score with:\n"
        f"    uv run python scripts/evaluate.py "
        f"--predictions {out_path} --eval_dir {eval_dir} --output results.json"
    )


if __name__ == "__main__":
    main()
