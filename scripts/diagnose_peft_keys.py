"""
diagnose_peft_keys.py
Loads `PaddlePaddle/PaddleOCR-VL` as the base, wraps it with peft using our
`adapter_config.json`, then compares the LoRA keys peft expects against what
`scripts/convert_paddleformers_to_peft.py` actually wrote into
`adapter_model.safetensors`.

This pinpoints exactly which path-prefix transformation the converter is
missing — usually a wrapped-attribute name like `language_model.` that peft
introduces on top of the underlying model.

Run inside the eval venv:

    cd /workspace/paddleocr-aksara-jawa
    uv run --extra eval python scripts/diagnose_peft_keys.py \\
        --adapter_dir /workspace/paddleocr-aksara-jawa/PaddleOCR-VL-Aksara-Jawa-lora/export

Apply the same nine compatibility patches `evaluate.py` does so the base
loads on transformers 5.6+; this script is read-only otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--adapter_dir", required=True,
                    help="Directory containing adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--base_model", default="PaddlePaddle/PaddleOCR-VL",
                    help="Base model id to wrap with peft (default: PaddlePaddle/PaddleOCR-VL)")
    ap.add_argument("--show", type=int, default=20,
                    help="How many sample keys to print from each list (default: 20)")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    adapter_cfg = adapter_dir / "adapter_config.json"
    adapter_safe = adapter_dir / "adapter_model.safetensors"
    if not adapter_cfg.exists():
        print(f"ERROR: {adapter_cfg} not found — run convert_paddleformers_to_peft.py first", file=sys.stderr)
        return 1
    if not adapter_safe.exists():
        print(f"ERROR: {adapter_safe} not found — run convert_paddleformers_to_peft.py first", file=sys.stderr)
        return 1

    try:
        import torch
        import transformers.modeling_utils as _t
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from transformers import AutoConfig, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from safetensors import safe_open
    except ImportError as e:
        print(f"ERROR: missing dep ({e}); run inside `uv run --extra eval`.", file=sys.stderr)
        return 1

    # ── Apply the same loading patches evaluate.py uses ───────────────────────
    _orig = _t.PreTrainedModel._init_weights
    def _safe_init(self, m):
        try:
            return _orig(self, m)
        except AttributeError as exc:
            if "compute_default_rope_parameters" in str(exc):
                return
            raise
    _t.PreTrainedModel._init_weights = _safe_init

    if "default" not in ROPE_INIT_FUNCTIONS:
        def _rope(config=None, device=None, seq_len=None):
            base = config.rope_theta
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            return (
                1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                .to(device=device, dtype=torch.float) / head_dim)),
                1.0,
            )
        ROPE_INIT_FUNCTIONS["default"] = _rope

    cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    mod_name = cfg.__class__.__module__.rsplit(".", 1)[0] + ".modeling_paddleocr_vl"
    mod = importlib.import_module(mod_name)
    if not hasattr(mod.RotaryEmbedding, "compute_default_rope_parameters"):
        mod.RotaryEmbedding.compute_default_rope_parameters = property(
            lambda self: self.rope_init_fn
        )

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # ── Build LoraConfig from our adapter_config.json ─────────────────────────
    with adapter_cfg.open() as f:
        ac = json.load(f)
    lc = LoraConfig(
        r=ac["r"],
        lora_alpha=ac["lora_alpha"],
        lora_dropout=ac.get("lora_dropout", 0.0),
        target_modules=ac["target_modules"],
        task_type=ac.get("task_type", "CAUSAL_LM"),
        bias=ac.get("bias", "none"),
    )
    print(f"Wrapping with peft (target_modules={ac['target_modules']})")
    peft_m = get_peft_model(base, lc)

    # ── Compare expected vs provided keys ─────────────────────────────────────
    expected = sorted(k for k in peft_m.state_dict().keys() if "lora" in k)
    with safe_open(str(adapter_safe), framework="pt") as f:
        provided = sorted(f.keys())

    print(f"\nPEFT EXPECTS {len(expected)} LoRA keys. First {args.show}:")
    for k in expected[:args.show]:
        print(f"  {k}")
    if len(expected) > args.show:
        print(f"  ... +{len(expected) - args.show} more")
    print(f"  Last 5 expected:")
    for k in expected[-5:]:
        print(f"    {k}")

    print(f"\nWE PROVIDED {len(provided)} keys. First 5:")
    for k in provided[:5]:
        print(f"  {k}")

    expected_set = set(expected)
    provided_set = set(provided)
    missing = expected_set - provided_set
    extra = provided_set - expected_set
    matched = expected_set & provided_set

    print()
    print(f"MATCHED            : {len(matched)} / {len(expected)}")
    print(f"PEFT EXPECTS, MISSING FROM US : {len(missing)}")
    print(f"WE PROVIDED, NOT EXPECTED     : {len(extra)}")

    if missing:
        print(f"\nSample MISSING-FROM-US (first {min(args.show, len(missing))}):")
        for k in sorted(missing)[:args.show]:
            print(f"  {k}")
    if extra:
        print(f"\nSample OUR-EXTRA (first {min(args.show, len(extra))}):")
        for k in sorted(extra)[:args.show]:
            print(f"  {k}")

    # Heuristic: spot a common prefix difference
    if missing and extra:
        m = sorted(missing)[0]
        e = sorted(extra)[0]
        print()
        print(f"For comparison, one missing key and one extra key in the same lexicographic position:")
        print(f"  MISSING: {m}")
        print(f"  EXTRA  : {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
