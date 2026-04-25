"""
evaluate.py
Evaluation script for PaddleOCR-VL Aksara Jawa fine-tuned model.

Computes Character Error Rate (CER), Word Error Rate (WER), and exact match
accuracy against the ground truth annotations in ground_truth.jsonl.

Uses the transformers + trust_remote_code pattern to load PaddleOCR-VL
(the VLM, not the classic det+rec pipeline) and jiwer for metric calculation.
Both reference and hypothesis are Unicode NFC-normalised before comparison,
so precomposed vs decomposed Javanese sequences don't score as different.

Usage:
    # Sanity-check ground truth only (no model inference)
    uv run python scripts/evaluate.py --gt_only --eval_dir data/eval/

    # Evaluate a model (requires the `eval` extras: uv sync --extra eval)
    uv run --extra eval python scripts/evaluate.py \\
        --model_path setilanaji/PaddleOCR-VL-Aksara-Jawa \\
        --eval_dir data/eval/ \\
        --output results.json

    # Evaluate from a predictions file (one prediction per line, matching GT order)
    uv run python scripts/evaluate.py \\
        --predictions predictions.txt \\
        --eval_dir data/eval/ \\
        --output results.json
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from jiwer import cer as jiwer_cer, wer as jiwer_wer


DEFAULT_PROMPT = "OCR:"


def normalize(s: str) -> str:
    """NFC-normalize and strip — applied to both hyp and ref before scoring."""
    return unicodedata.normalize("NFC", s or "").strip()


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate. jiwer.cer handles empty-ref edge cases correctly."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer_cer(ref, hyp))


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref.split():
        return 0.0 if not hyp.split() else 1.0
    return float(jiwer_wer(ref, hyp))


# ── Pasangan (stacked-consonant) metric ───────────────────────────────────────
#
# Aksara Jawa doesn't have a dedicated pasangan codepoint — a pasangan cluster
# is encoded as Consonant + PANGKON (U+A9C0, virama) + Consonant. The renderer
# produces the stacked form. We count these clusters in ref vs hyp as a direct
# measure of how well the model handles the script's most visually complex
# element.

PANGKON = "꧀"
# Javanese letters (carakan + murda + additional letters), per Unicode block
# allocation in U+A984–U+A9B2. Excludes sandhangan (marks) and punctuation.
_JAV_LETTER_RANGE = range(0xA984, 0xA9B3)


def _is_jav_letter(c: str) -> bool:
    return bool(c) and ord(c) in _JAV_LETTER_RANGE


def pasangan_clusters(s: str) -> list[str]:
    """All C + pangkon + C triples in s, preserving duplicates."""
    out: list[str] = []
    s = normalize(s)
    for i in range(len(s) - 2):
        if (
            s[i + 1] == PANGKON
            and _is_jav_letter(s[i])
            and _is_jav_letter(s[i + 2])
        ):
            out.append(s[i : i + 3])
    return out


def pasangan_metrics(hypothesis: str, reference: str) -> dict:
    """
    Return {ref_count, hyp_count, correct, recall, precision, f1} for pasangan
    clusters. Multi-set intersection: counts each distinct triple by how many
    times it appears in both strings.
    """
    from collections import Counter

    ref_c = Counter(pasangan_clusters(reference))
    hyp_c = Counter(pasangan_clusters(hypothesis))
    ref_n = sum(ref_c.values())
    hyp_n = sum(hyp_c.values())
    correct = sum((ref_c & hyp_c).values())

    recall = correct / ref_n if ref_n else (1.0 if hyp_n == 0 else 0.0)
    precision = correct / hyp_n if hyp_n else (1.0 if ref_n == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "ref_count": ref_n,
        "hyp_count": hyp_n,
        "correct": correct,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


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
            "image":       obj["image"],
            "text":        text,
            "script_type": obj.get("script_type") or "unspecified",
        })
    return records


# ── Model inference ───────────────────────────────────────────────────────────

def run_model_inference(
    model_path: str,
    eval_dir: Path,
    ground_truth: list[dict],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    peft_adapter: str | None = None,
) -> list[str]:
    """
    Run PaddleOCR-VL inference on eval images via the HuggingFace transformers
    interface with trust_remote_code. Returns predictions in ground_truth order.

    If `peft_adapter` is provided, `model_path` is treated as the BASE model
    and the adapter directory's LoRA weights are applied at runtime via
    `peft.PeftModel.from_pretrained`. This skips the paddleformers-cli export
    merge step that was discovered to corrupt the merged safetensors output.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError as e:
        print(
            f"ERROR: inference dependencies missing ({e}). Install with:\n"
            f"    uv sync --extra eval"
        )
        sys.exit(1)

    # transformers 5.x's PreTrainedModel._init_weights calls
    # `module.compute_default_rope_parameters` on rotary embedding modules
    # during its post-load weight initialization pass. PaddleOCR-VL's
    # RotaryEmbedding stores the function as `rope_init_fn` instead, and
    # _init_weights blows up with AttributeError. The weights are *already*
    # loaded (we have no actually-missing keys) — this is just defensive
    # re-init for would-be-missing modules. Wrap _init_weights to swallow
    # this specific failure; everything else propagates normally.
    import transformers.modeling_utils as _tmu
    if not getattr(_tmu.PreTrainedModel._init_weights, "_aksara_patched", False):
        _orig_init_weights = _tmu.PreTrainedModel._init_weights
        def _safe_init_weights(self, module):
            try:
                return _orig_init_weights(self, module)
            except AttributeError as exc:
                if "compute_default_rope_parameters" in str(exc):
                    return  # weights loaded fine; rope is already correct
                raise
        _safe_init_weights._aksara_patched = True
        _tmu.PreTrainedModel._init_weights = _safe_init_weights

    # PaddleOCR-VL's modeling code references ROPE_INIT_FUNCTIONS["default"],
    # which transformers removed in 5.0. Re-register the 4.57.x implementation
    # so the model can initialise its RotaryEmbedding without crashing.
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config=None, device=None, seq_len=None):
            base = config.rope_theta
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                         .to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0  # (inv_freq, attention_factor)
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    print(f"Loading model from: {model_path}")

    # transformers 5.x's _init_weights calls `module.compute_default_rope_parameters`
    # on the model's RotaryEmbedding instance, but PaddleOCR-VL's RotaryEmbedding
    # stores the rope-init function as `rope_init_fn` instead. Trigger the trust-
    # remote-code load via AutoConfig, then patch the class before instantiation.
    from transformers import AutoConfig
    import importlib
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    modeling_mod_name = cfg.__class__.__module__.rsplit(".", 1)[0] + ".modeling_paddleocr_vl"
    try:
        modeling_mod = importlib.import_module(modeling_mod_name)
        if hasattr(modeling_mod, "RotaryEmbedding"):
            re_cls = modeling_mod.RotaryEmbedding
            if not hasattr(re_cls, "compute_default_rope_parameters"):
                # Make the attribute resolvable on instances by routing it to
                # whatever rope_init_fn each instance was set up with at __init__.
                re_cls.compute_default_rope_parameters = property(lambda self: self.rope_init_fn)
                print("  patched RotaryEmbedding.compute_default_rope_parameters → rope_init_fn")
    except Exception as e:
        print(f"  WARNING: could not patch RotaryEmbedding ({e}); model load may fail")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # IMPORTANT: patch cache_position on the BASE model class BEFORE peft
    # wraps it. After PeftModel.from_pretrained, type(model) becomes
    # PeftModelForCausalLM and our patch on it would never be reached —
    # peft.generate routes back through the underlying class's prepare_inputs_for_generation.
    _base_cls = type(model)
    if not getattr(_base_cls.prepare_inputs_for_generation, "_aksara_patched", False):
        _orig_base_prep = _base_cls.prepare_inputs_for_generation
        def _base_prep_with_cache_position(self, input_ids, past_key_values=None, **kwargs):
            if kwargs.get("cache_position") is None:
                past_seen = 0
                if past_key_values is not None:
                    try:
                        past_seen = past_key_values.get_seq_length()
                    except (AttributeError, TypeError):
                        try:
                            past_seen = past_key_values[0][0].shape[-2]
                        except Exception:
                            past_seen = 0
                cur_len = input_ids.shape[1]
                kwargs["cache_position"] = torch.arange(past_seen, cur_len, device=input_ids.device)
            return _orig_base_prep(self, input_ids, past_key_values=past_key_values, **kwargs)
        _base_prep_with_cache_position._aksara_patched = True
        _base_cls.prepare_inputs_for_generation = _base_prep_with_cache_position

    if peft_adapter:
        # Apply LoRA at runtime instead of using the broken merged safetensors.
        # paddleformers-cli writes the adapter as `peft_model-*.safetensors` +
        # `lora_config.json`; HF peft expects `adapter_model.safetensors` +
        # `adapter_config.json`. If those names aren't found, peft will raise
        # — point the user at convert_paddleformers_lora.py (TBD) or symlink.
        try:
            from peft import PeftModel
        except ImportError:
            print("ERROR: peft not installed. Add it to the eval extra:")
            print("    uv pip install peft")
            sys.exit(1)
        print(f"  Applying LoRA adapter from: {peft_adapter}")
        model = PeftModel.from_pretrained(model, peft_adapter).eval()
        print(f"  LoRA loaded; trainable params now: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # transformers 5.6+ no longer auto-creates `cache_position` for remote-code
    # models, but PaddleOCR-VL's prepare_inputs_for_generation still does
    # `if cache_position[0] != 0:` and crashes with NoneType subscript. Wrap
    # the method to fill in the position vector that transformers used to pass.
    _model_cls = type(model)
    if not getattr(_model_cls.prepare_inputs_for_generation, "_aksara_patched", False):
        _orig_prep = _model_cls.prepare_inputs_for_generation
        def _prep_with_cache_position(self, input_ids, past_key_values=None, **kwargs):
            if kwargs.get("cache_position") is None:
                past_seen = 0
                if past_key_values is not None:
                    try:
                        past_seen = past_key_values.get_seq_length()
                    except (AttributeError, TypeError):
                        try:
                            past_seen = past_key_values[0][0].shape[-2]
                        except Exception:
                            past_seen = 0
                cur_len = input_ids.shape[1]
                kwargs["cache_position"] = torch.arange(
                    past_seen, past_seen + cur_len - past_seen,
                    device=input_ids.device,
                ) if cur_len > past_seen else torch.arange(
                    past_seen, cur_len, device=input_ids.device,
                )
                # Simpler: just cover positions [past_seen, cur_len)
                kwargs["cache_position"] = torch.arange(
                    past_seen, cur_len, device=input_ids.device,
                )
            return _orig_prep(self, input_ids, past_key_values=past_key_values, **kwargs)
        _prep_with_cache_position._aksara_patched = True
        _model_cls.prepare_inputs_for_generation = _prep_with_cache_position

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    predictions: list[str] = []
    for i, record in enumerate(ground_truth):
        img_path = eval_dir / record["image"]
        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            predictions.append("")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            # Multimodal-content list IS the right format here — apply_chat_template
            # needs to know there's an image so it injects the image placeholder
            # tokens into the prompt. Plain-string content makes the template emit
            # zero image tokens and the model errors with "tokens: 0, features N".
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {
                k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            input_length = inputs["input_ids"].shape[1]
            pred = processor.batch_decode(
                generated[:, input_length:], skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            print(f"  WARNING: inference failed for {record['image']}: {e}")
            if i == 0:
                # Print the full traceback for the first failure so we can
                # diagnose root cause; subsequent failures stay terse.
                import traceback as _tb
                _tb.print_exc()
            pred = ""

        predictions.append(pred)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1:>4}/{len(ground_truth)}] {record['image']}")

    return predictions


# ── Evaluation ────────────────────────────────────────────────────────────────

def _aggregate(records: list[dict]) -> dict:
    """Aggregate per-image records into overall + pasangan + counts."""
    if not records:
        return {
            "total_images":  0,
            "mean_cer":      0.0,
            "mean_wer":      0.0,
            "exact_matches": 0,
            "exact_rate":    0.0,
            "pasangan":      {"ref_count": 0, "correct": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0},
        }
    n = len(records)
    mean_cer = sum(r["cer"] for r in records) / n
    mean_wer = sum(r["wer"] for r in records) / n
    exact_matches = sum(1 for r in records if r["exact_match"])

    # Pasangan aggregated across all records — sum counts, then recompute
    pas_ref = sum(r["pasangan"]["ref_count"] for r in records)
    pas_hyp = sum(r["pasangan"]["hyp_count"] for r in records)
    pas_correct = sum(r["pasangan"]["correct"] for r in records)
    recall = pas_correct / pas_ref if pas_ref else (1.0 if pas_hyp == 0 else 0.0)
    precision = pas_correct / pas_hyp if pas_hyp else (1.0 if pas_ref == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total_images":  n,
        "mean_cer":      round(mean_cer, 4),
        "mean_wer":      round(mean_wer, 4),
        "exact_matches": exact_matches,
        "exact_rate":    round(exact_matches / n, 4),
        "pasangan": {
            "ref_count": pas_ref,
            "hyp_count": pas_hyp,
            "correct":   pas_correct,
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "f1":        round(f1, 4),
        },
    }


def evaluate(
    predictions: list[str],
    ground_truth: list[dict],
    verbose: bool = False,
) -> dict:
    """
    Compute CER, WER, exact match, pasangan recall/precision, and per-tier
    aggregates (printed/handwritten/manuscript/unspecified).
    """
    assert len(predictions) == len(ground_truth), \
        f"Prediction count ({len(predictions)}) != ground truth count ({len(ground_truth)})"

    per_image: list[dict] = []
    for pred, gt in zip(predictions, ground_truth):
        ref = normalize(gt["text"])
        hyp = normalize(pred)

        c = cer(hyp, ref)
        w = wer(hyp, ref)
        exact = (hyp == ref)
        pas = pasangan_metrics(hyp, ref)

        per_image.append({
            "image":       gt["image"],
            "script_type": gt.get("script_type", "unspecified"),
            "reference":   ref,
            "hypothesis":  hyp,
            "cer":         round(c, 4),
            "wer":         round(w, 4),
            "exact_match": exact,
            "pasangan":    pas,
        })

        if verbose and not exact:
            print(f"  MISS  {gt['image']}  ({gt.get('script_type', 'unspecified')})")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")
            print(f"    CER: {c:.4f}  WER: {w:.4f}  pasangan F1: {pas['f1']:.4f}")

    overall = _aggregate(per_image)

    # Per-tier breakdown
    tiers: dict[str, list[dict]] = {}
    for r in per_image:
        tiers.setdefault(r["script_type"], []).append(r)
    per_tier = {tier: _aggregate(recs) for tier, recs in tiers.items()}

    return {
        **overall,
        "per_tier":  per_tier,
        "per_image": per_image,
    }


def diff_results(fine_tuned: dict, baseline: dict) -> dict:
    """Build a comparison summary: baseline vs fine-tuned, overall + per-tier."""
    def _diff(a: dict, b: dict) -> dict:
        return {
            "cer_baseline":     b["mean_cer"],
            "cer_fine_tuned":   a["mean_cer"],
            "cer_delta":        round(a["mean_cer"] - b["mean_cer"], 4),
            "wer_baseline":     b["mean_wer"],
            "wer_fine_tuned":   a["mean_wer"],
            "wer_delta":        round(a["mean_wer"] - b["mean_wer"], 4),
            "exact_baseline":   b["exact_rate"],
            "exact_fine_tuned": a["exact_rate"],
            "exact_delta":      round(a["exact_rate"] - b["exact_rate"], 4),
            "pasangan_f1_baseline":   b["pasangan"]["f1"],
            "pasangan_f1_fine_tuned": a["pasangan"]["f1"],
            "pasangan_f1_delta":      round(a["pasangan"]["f1"] - b["pasangan"]["f1"], 4),
        }

    result: dict = {"overall": _diff(fine_tuned, baseline), "per_tier": {}}
    shared_tiers = set(fine_tuned.get("per_tier", {})) & set(baseline.get("per_tier", {}))
    for tier in sorted(shared_tiers):
        result["per_tier"][tier] = _diff(
            fine_tuned["per_tier"][tier], baseline["per_tier"][tier]
        )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_summary(results: dict, label: str) -> None:
    pas = results.get("pasangan", {})
    print(f"  {label}")
    print(f"    Images          : {results['total_images']}")
    print(f"    Mean CER        : {results['mean_cer']:.4f} ({results['mean_cer']*100:.2f}%)")
    print(f"    Mean WER        : {results['mean_wer']:.4f} ({results['mean_wer']*100:.2f}%)")
    print(f"    Exact matches   : {results['exact_matches']} / {results['total_images']} ({results['exact_rate']*100:.2f}%)")
    if pas.get("ref_count"):
        print(
            f"    Pasangan        : recall={pas['recall']:.4f}  "
            f"precision={pas['precision']:.4f}  F1={pas['f1']:.4f}  "
            f"(ref={pas['ref_count']} hyp={pas['hyp_count']} correct={pas['correct']})"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Aksara Jawa OCR model using CER, WER, and exact match.")
    ap.add_argument("--eval_dir",            type=str, default="data/eval/",
                    help="Directory containing ground_truth.jsonl and images")
    ap.add_argument("--model_path",          type=str, default=None,
                    help="Path or HF repo id of the fine-tuned PaddleOCR-VL model")
    ap.add_argument("--baseline_model_path", type=str, default=None,
                    help="Second model to compare against (e.g. PaddlePaddle/PaddleOCR-VL for the base model)")
    ap.add_argument("--predictions",         type=str, default=None,
                    help="Path to predictions file (one prediction per line)")
    ap.add_argument("--output",              type=str, default="results.json",
                    help="Output path for results JSON")
    ap.add_argument("--gt_only",             action="store_true",
                    help="Check ground truth only (no inference) — for validation")
    ap.add_argument("--verbose",             action="store_true",
                    help="Print mismatches during evaluation")
    ap.add_argument("--top_k",               type=int, default=10,
                    help="Number of worst-performing images to print (default: 10)")
    ap.add_argument("--prompt",              type=str, default=DEFAULT_PROMPT,
                    help="OCR prompt sent to the model (default: 'OCR:')")
    ap.add_argument("--peft_adapter",        type=str, default=None,
                    help="Optional path to a LoRA adapter dir to apply on top of --model_path "
                         "at runtime via peft.PeftModel.from_pretrained. Use this to bypass "
                         "the broken paddleformers-cli merged safetensors.")
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
        predictions = run_model_inference(
            args.model_path, eval_dir, ground_truth,
            prompt=args.prompt, peft_adapter=args.peft_adapter,
        )

    else:
        print("ERROR: provide one of --gt_only, --predictions, or --model_path")
        ap.print_help()
        sys.exit(1)

    # Evaluate
    print("\nComputing metrics...")
    results = evaluate(predictions, ground_truth, verbose=args.verbose)

    # Optional baseline run
    baseline_results = None
    if args.baseline_model_path:
        print(f"\nRunning baseline inference: {args.baseline_model_path}")
        baseline_preds = run_model_inference(
            args.baseline_model_path, eval_dir, ground_truth, prompt=args.prompt
        )
        baseline_results = evaluate(baseline_preds, ground_truth, verbose=False)
        results["baseline"] = baseline_results
        results["diff"] = diff_results(results, baseline_results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS" + (" (fine-tuned vs baseline)" if baseline_results else ""))
    print("=" * 60)
    _print_summary(results, label="Fine-tuned" if baseline_results else "Model")

    if baseline_results:
        print()
        _print_summary(baseline_results, label="Baseline")
        print()
        print("-" * 60)
        print("DELTA (fine-tuned − baseline)")
        print("-" * 60)
        d = results["diff"]["overall"]
        print(f"  ΔCER            : {d['cer_delta']:+.4f}  ({d['cer_baseline']:.4f} → {d['cer_fine_tuned']:.4f})")
        print(f"  ΔWER            : {d['wer_delta']:+.4f}  ({d['wer_baseline']:.4f} → {d['wer_fine_tuned']:.4f})")
        print(f"  Δexact rate     : {d['exact_delta']:+.4f}  ({d['exact_baseline']:.4f} → {d['exact_fine_tuned']:.4f})")
        print(f"  Δpasangan F1    : {d['pasangan_f1_delta']:+.4f}  ({d['pasangan_f1_baseline']:.4f} → {d['pasangan_f1_fine_tuned']:.4f})")
        if results["diff"]["per_tier"]:
            print("\n  Per-tier ΔCER:")
            for tier, dt in sorted(results["diff"]["per_tier"].items()):
                print(f"    {tier:<15}  {dt['cer_delta']:+.4f}  ({dt['cer_baseline']:.4f} → {dt['cer_fine_tuned']:.4f})")

    # Per-tier breakdown
    if results.get("per_tier") and len(results["per_tier"]) > 1:
        print("\n" + "-" * 60)
        print("PER-TIER BREAKDOWN")
        print("-" * 60)
        for tier in sorted(results["per_tier"]):
            t = results["per_tier"][tier]
            print(f"  {tier:<15}  n={t['total_images']:>3}  "
                  f"CER={t['mean_cer']:.4f}  WER={t['mean_wer']:.4f}  "
                  f"exact={t['exact_rate']:.4f}  pasangan-F1={t['pasangan']['f1']:.4f}")

    # Print worst-performing images
    if not args.gt_only and args.top_k > 0:
        worst = sorted(results["per_image"], key=lambda x: x["cer"], reverse=True)[:args.top_k]
        print(f"\nTop {args.top_k} worst CER images:")
        for item in worst:
            print(f"  {item['image']}  CER={item['cer']:.4f}  WER={item['wer']:.4f}  "
                  f"tier={item.get('script_type', '?')}")
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