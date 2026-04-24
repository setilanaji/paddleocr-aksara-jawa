"""
hf_push_dataset.py
Assemble and upload the Aksara Jawa OCR dataset to Hugging Face, mirroring
the structure documented in docs/data_report.md §6.

Layout on HF (matches canonical local paths):

    setilanaji/aksara-jawa-ocr/
      synthetic/              2,000 images + ground_truth.jsonl + Label.txt
      semi_synthetic/         1,000 images + ground_truth.jsonl + Label.txt
      real/                     800 images + sources.csv  (Leiden, PDM 1.0)
      eval/
        synthetic_v1/           150 synthetic eval images
        real_v2/                annotations + candidate list (added once
                                labelling is complete)
      training/
        ocr_vl_sft-train_aksara_jawa.jsonl
        ocr_vl_sft-test_aksara_jawa.jsonl
      README.md                dataset card with per-subset license block

All tiers are currently redistributable:
  - synthetic/ + semi_synthetic/ + eval/synthetic_v1/ render Noto Sans
    Javanese glyphs under SIL OFL 1.1 (permits redistribution).
  - real/ is 800 Leiden Or. 1871 + Or. 1928 + D Or. 15 pages under Public
    Domain Mark 1.0; the Leiden attribution string is reproduced in the card.

Usage:
    # Push everything (the big first-time push)
    uv run python scripts/hf_push_dataset.py

    # Dry-run — print what would upload, transfer nothing
    uv run python scripts/hf_push_dataset.py --dry-run

    # Only push one tier (faster iteration during card/layout changes)
    uv run python scripts/hf_push_dataset.py --only synthetic
    uv run python scripts/hf_push_dataset.py --only readme

    # Different repo (for forks / testing)
    uv run python scripts/hf_push_dataset.py --repo myuser/aksara-jawa-ocr

Auth: expects HF_TOKEN in env, or prior `huggingface-cli login`.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "setilanaji/aksara-jawa-ocr"
REPO_TYPE = "dataset"

# Tier name → (local folder, path_in_repo, allow_patterns, optional=False)
# Absent tiers (e.g. eval/real_v2/ before annotation completes) are skipped
# with a note, not an error.
TIERS: list[tuple[str, str, str, list[str], bool]] = [
    (
        "synthetic",
        "data/synthetic",
        "synthetic",
        ["*.jpg", "ground_truth.jsonl", "Label.txt", "dataset_stats.json", "ocr_vl_sft.jsonl"],
        False,
    ),
    (
        "semi_synthetic",
        "data/semi_synthetic",
        "semi_synthetic",
        ["*.jpg", "ground_truth.jsonl", "Label.txt", "dataset_stats.json", "ocr_vl_sft.jsonl"],
        False,
    ),
    (
        "real",
        "data/real",
        "real",
        ["*.jpg", "sources.csv"],
        False,
    ),
    (
        "eval_synthetic_v1",
        "data/eval",
        "eval/synthetic_v1",
        ["*.jpg", "ground_truth.jsonl", "Label.txt", "dataset_stats.json", "ocr_vl_sft.jsonl"],
        False,
    ),
    (
        "eval_candidates",
        "data",
        "eval",
        ["eval_real_candidates.txt"],
        False,
    ),
    (
        "training",
        "training",
        "training",
        ["ocr_vl_sft-train_aksara_jawa.jsonl", "ocr_vl_sft-test_aksara_jawa.jsonl"],
        False,
    ),
    (
        "eval_real_v2",
        "data/real",
        "eval/real_v2",
        ["ground_truth.jsonl", "Label.txt"],
        True,  # exists only after annotation is exported
    ),
]


DATASET_CARD = """\
---
language:
- jv
license: other
task_categories:
- image-to-text
pretty_name: Aksara Jawa OCR
size_categories:
- 1K<n<10K
tags:
- ocr
- javanese
- aksara-jawa
- manuscript
- handwritten
- historical
- paddleocr
- paddleocr-vl
---

# Aksara Jawa OCR Dataset

Training and evaluation data for fine-tuning [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) on **Aksara Jawa (Javanese script, U+A980–U+A9DF)**.

Part of the [PaddleOCR 10th Hackathon — Derivative Model Challenge](https://github.com/PaddlePaddle/PaddleOCR/issues/17858).

- **Model:** [setilanaji/PaddleOCR-VL-Aksara-Jawa](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa)
- **Source code:** [github.com/setilanaji/paddleocr-aksara-jawa](https://github.com/setilanaji/paddleocr-aksara-jawa)
- **Training data construction report:** [`docs/data_report.md`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/docs/data_report.md)

## Contents

| Subset | Count | License | Description |
|---|---:|---|---|
| `synthetic/` | 2,000 | [SIL OFL 1.1](https://openfontlicense.org/) | Pure synthetic text — Noto Sans Javanese rendered on solid / aged-paper backgrounds. Three augmentation tiers (light / medium / heavy). 20% pasangan-stress samples from a dedicated stacked-consonant corpus. |
| `semi_synthetic/` | 1,000 | SIL OFL 1.1 (text) + [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) (backgrounds) | Synthetic text rendered on heavily blurred real Leiden manuscript backgrounds (σ=8–16 Gaussian blur, random crop). Teaches domain-adaptation texture without requiring transcription. **Excludes the 120 eval-candidate pages so training and eval stay disjoint.** |
| `real/` | 800 | Public Domain Mark 1.0 | Leiden University Libraries manuscript pages — Or. 1871 (640 pages, *Panji Jaya Lengkara and Angreni* romance), Or. 1928 (42 pages, *Kitab pangeran Bonang*, dluwang paper), D Or. 15 (118 pages, *Babad Paku Alaman*, illuminated court manuscript). All Leiden items are declared Public Domain Mark 1.0 on the source pages. |
| `eval/synthetic_v1/` | 150 | SIL OFL 1.1 | Synthetic eval set (light augmentation only, seed=42). Baseline for CER/WER sanity checking. |
| `eval/real_v2/` | ~100 | PDM 1.0 (images) + CC0 (annotations) | Line-level Unicode Javanese transcriptions of a stratified subset of `real/` — 80 Or. 1871 + 30 D Or. 15 + 10 Or. 1928, picked by `scripts/pick_eval_candidates.py` (seed=42). Ground truth is annotated in Label Studio and exported to PaddleOCR `Label.txt` + conversation `ground_truth.jsonl` formats. *Published when annotation completes.* |
| `eval/eval_real_candidates.txt` | — | CC0 | The pinned, deterministic list of `real/` filenames reserved for eval. Regeneratable via `uv run python scripts/pick_eval_candidates.py`. |
| `training/` | — | Same as above, mixed | Merged `ocr_vl_sft-train_aksara_jawa.jsonl` (3,000 records = 2,000 `synthetic/` + 1,000 `semi_synthetic/`) and `ocr_vl_sft-test_aksara_jawa.jsonl` (150 `eval/synthetic_v1/` records). PaddleFormers `messages` + `images` SFT schema. |

## Attribution

### Leiden manuscripts (`real/`, `eval/real_v2/`)

Images are reproduced from [Leiden University Libraries — Digital Collections](https://digitalcollections.universiteitleiden.nl/) under Public Domain Mark 1.0. Per-image provenance (source URL, IIIF manifest, canvas label) is tracked in `real/sources.csv`.

> Citing the Leiden University Libraries as a source is appreciated.

Shelfmarks included: **Or. 1871** (Persistent URL: `http://hdl.handle.net/1887.1/item:1990266`), **Or. 1928**, **D Or. 15**. Pages harvested at 2585-wide resolution to ensure sandhangan diacritics resolve at 15–20 px.

### Noto Sans Javanese font (`synthetic/`, `semi_synthetic/`, `eval/synthetic_v1/`)

Typefaces used for synthetic rendering are Noto Sans Javanese Regular + Bold, distributed under the [SIL Open Font License 1.1](https://openfontlicense.org/), which permits embedding in generated images and redistribution.

## Reproducibility

All pipelines are seed-pinned:

- `scripts/generate_aksara.py` — `seed=42` for `synthetic/`, `seed=142` for `semi_synthetic/`, `seed=42` (via `--eval`) for `eval/synthetic_v1/`.
- `scripts/pick_eval_candidates.py` — `seed=42` produces the 120-file candidate list.
- `scripts/convert_format.py --shuffle` — derives the merged training JSONL.

Full build recipe: see §11 of [`docs/data_report.md`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/docs/data_report.md).

## Intended use

Fine-tuning and evaluating OCR / document-parsing models on Aksara Jawa. Handwritten / palm-leaf manuscript recognition is a known gap in public datasets — this release is specifically aimed at filling it.

## Limitations

See §10 of [`docs/data_report.md`](https://github.com/setilanaji/paddleocr-aksara-jawa/blob/main/docs/data_report.md). Key items: single-font synthetic rendering (Noto Sans Javanese only), corpus size (303 handcrafted lines + 150 pasangan-stress lines), no Yogyakarta street-signage tier in the eval set yet.
"""


def expand_allow_patterns(base: Path, patterns: list[str]) -> list[Path]:
    """Resolve allow_patterns against a local folder for counting / dry-run."""
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(sorted(base.glob(pat)))
    return hits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF dataset repo ID (default: {DEFAULT_REPO})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan; upload nothing")
    ap.add_argument("--only",
                    help="Push only one tier name (synthetic, semi_synthetic, real, "
                         "eval_synthetic_v1, eval_candidates, training, eval_real_v2, readme). "
                         "Use when iterating on the card or one tier.")
    args = ap.parse_args()

    root = Path(".").resolve()
    api = HfApi()

    # Plan
    plan: list[tuple[str, Path, str, list[str], int]] = []
    for name, folder, path_in_repo, patterns, optional in TIERS:
        if args.only and args.only not in (name, "readme"):
            continue
        if args.only == "readme":
            continue
        base = root / folder
        if not base.exists():
            if optional:
                print(f"  SKIP {name}: {folder} does not exist yet (optional tier)")
                continue
            raise SystemExit(f"required tier {name!r} source missing: {folder}")
        files = expand_allow_patterns(base, patterns)
        if not files:
            if optional:
                print(f"  SKIP {name}: no files match patterns {patterns} in {folder}")
                continue
            raise SystemExit(
                f"required tier {name!r} has no files matching {patterns} in {folder}"
            )
        plan.append((name, base, path_in_repo, patterns, len(files)))

    # Report plan
    print(f"Target repo: {args.repo}  (type={REPO_TYPE})")
    if not plan and args.only != "readme":
        print("Nothing to upload.")
        return
    for name, base, path_in_repo, patterns, n in plan:
        print(f"  [{name:<18}] {n:>5} files  {base}  →  {path_in_repo}/  "
              f"patterns={patterns}")
    if args.only in (None, "readme"):
        print(f"  [{'readme':<18}] dataset card  →  README.md")
    if args.dry_run:
        print("\n--dry-run set; nothing uploaded.")
        return

    # Make sure repo exists
    api.create_repo(args.repo, repo_type=REPO_TYPE, exist_ok=True)

    # Push each tier
    for name, base, path_in_repo, patterns, _ in plan:
        print(f"\n↑ uploading {name} ({base} → {path_in_repo}/)")
        api.upload_folder(
            repo_id=args.repo,
            repo_type=REPO_TYPE,
            folder_path=str(base),
            path_in_repo=path_in_repo,
            allow_patterns=patterns,
            commit_message=f"push tier: {name}",
        )

    # Push README
    if args.only in (None, "readme"):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".md", delete=False
        ) as f:
            f.write(DATASET_CARD)
            card_path = f.name
        try:
            print("\n↑ uploading README.md (dataset card)")
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type=REPO_TYPE,
                commit_message="update dataset card",
            )
        finally:
            os.unlink(card_path)

    print(f"\nDone. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
