"""
convert_paddleformers_to_peft.py
Translate paddleformers' LoRA adapter dump (`lora_config.json` +
`peft_model-*.safetensors`) into the HuggingFace `peft` library's expected
schema (`adapter_config.json` + `adapter_model.safetensors`) IN PLACE.

This sidesteps the `paddleformers-cli export` merge step that — at least in
the v1 weights as exported in 2025-10 — produces merged safetensors whose
forward pass emits invalid byte sequences. Loading the unmerged adapter on
top of the unmodified base via `peft.PeftModel.from_pretrained` should
preserve the trained LoRA's actual learned values.

Usage:
    python3 scripts/convert_paddleformers_to_peft.py \\
        /path/to/PaddleOCR-VL-Aksara-Jawa-lora/export

After this writes the adapter config + symlinks the safetensors, run:
    uv run --extra eval python scripts/evaluate.py \\
        --model_path PaddlePaddle/PaddleOCR-VL \\
        --peft_adapter /path/to/...export \\
        --baseline_model_path PaddlePaddle/PaddleOCR-VL \\
        --eval_dir data/eval/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def translate_target_modules(patterns: list[str]) -> list[str]:
    """
    paddleformers writes regex like `.*model.*q_proj.*`. HF peft accepts
    `target_modules` as a list of *suffix* substrings (matches any module
    name ending in one). Strip the `.*` wrappers and any `.*` infixes, keep
    the trailing module name.
    """
    out: set[str] = set()
    for p in patterns:
        core = p.strip(".*").strip()
        if ".*" in core:
            core = core.split(".*")[-1]
        core = core.strip(".*").strip()
        if core:
            out.add(core)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("adapter_dir", help="Directory containing lora_config.json + peft_model-*.safetensors")
    ap.add_argument("--task_type", default="CAUSAL_LM",
                    help="HF peft task_type (default: CAUSAL_LM; for some VLM heads try FEATURE_EXTRACTION)")
    ap.add_argument("--inspect_keys", action="store_true",
                    help="Print the first 20 weight keys from peft_model-*.safetensors for debugging")
    args = ap.parse_args()

    d = Path(args.adapter_dir).resolve()
    if not d.is_dir():
        print(f"ERROR: not a directory: {d}", file=sys.stderr)
        return 1

    pf_cfg_path = d / "lora_config.json"
    if not pf_cfg_path.exists():
        print(f"ERROR: {pf_cfg_path} not found — is this a paddleformers LoRA export?", file=sys.stderr)
        return 1

    with pf_cfg_path.open() as f:
        pf = json.load(f)

    # Build the HF peft schema
    target_modules = translate_target_modules(pf.get("target_modules", []))
    adapter_cfg = {
        "auto_mapping": None,
        "base_model_name_or_path": pf.get("base_model_name_or_path"),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": pf.get("lora_alpha", 32),
        "lora_dropout": pf.get("lora_dropout", 0.0),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": pf.get("r", 16),
        "revision": None,
        "target_modules": target_modules,
        "task_type": args.task_type,
    }

    out_cfg = d / "adapter_config.json"
    with out_cfg.open("w") as f:
        json.dump(adapter_cfg, f, indent=2)
    print(f"  wrote {out_cfg.relative_to(d)}")
    print(f"    target_modules ({len(target_modules)}): {target_modules}")
    print(f"    r={adapter_cfg['r']}  lora_alpha={adapter_cfg['lora_alpha']}  task_type={adapter_cfg['task_type']}")

    # Find peft_model-*.safetensors and link as adapter_model.safetensors.
    pf_shards = sorted(d.glob("peft_model-*.safetensors"))
    if not pf_shards:
        print(f"ERROR: no peft_model-*.safetensors found in {d}", file=sys.stderr)
        return 1

    # Rewrite keys to match HF peft's expected schema:
    #   paddleformers: `mlp_AR.linear_1.lora_A`
    #   HF peft:      `base_model.model.mlp_AR.linear_1.lora_A.default.weight`
    # peft loads the LoRA into a wrapped model where the underlying base is
    # accessed at `base_model.model.<original_path>`, and each LoRA module
    # has its parameters under the `.default` adapter slot, with `.weight`
    # at the end (since LoRA matrices are nn.Parameter on a Linear module).
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        print("ERROR: safetensors[torch] not installed; cannot rewrite keys", file=sys.stderr)
        return 1

    out_safe = d / "adapter_model.safetensors"
    if out_safe.exists() or out_safe.is_symlink():
        out_safe.unlink()

    if len(pf_shards) > 1:
        print(f"WARNING: {len(pf_shards)} shards — concatenating into a single adapter_model.safetensors", file=sys.stderr)

    merged: dict[str, "torch.Tensor"] = {}
    for shard in pf_shards:
        merged.update(load_file(str(shard)))

    new_state = {}
    for k, v in merged.items():
        nk = "base_model.model." + k
        if nk.endswith(".lora_A") or nk.endswith(".lora_B"):
            nk = nk + ".default.weight"
        new_state[nk] = v

    save_file(new_state, str(out_safe))
    print(f"  wrote {out_safe.name}: {len(new_state)} keys, prefixed `base_model.model.` and `.default.weight` suffix")
    print(f"    sample keys:")
    for k in list(new_state.keys())[:3]:
        print(f"      {k}")

    if args.inspect_keys:
        try:
            from safetensors import safe_open
        except ImportError:
            print("(install safetensors to use --inspect_keys)")
        else:
            with safe_open(str(pf_shards[0]), framework="pt") as f:
                keys = list(f.keys())
            print(f"\n  safetensors keys: {len(keys)} total")
            for k in keys[:20]:
                print(f"    {k}")
            if len(keys) > 20:
                print(f"    ... +{len(keys) - 20} more")

    print("\nDone. Next:")
    print(f"  uv run --extra eval python scripts/evaluate.py \\")
    print(f"      --model_path PaddlePaddle/PaddleOCR-VL \\")
    print(f"      --peft_adapter {d} \\")
    print(f"      --baseline_model_path PaddlePaddle/PaddleOCR-VL \\")
    print(f"      --eval_dir data/eval/ --output results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
