"""
annotator_agreement.py
Computes inter-annotator agreement for Aksara Jawa line transcriptions.

Takes two Label Studio JSON exports (same project, two different annotators)
and reports the mean Character Error Rate (CER) between their transcriptions
for images both annotated. Both sides are Unicode NFC-normalised before
comparison so precomposed vs decomposed sequences don't score as different.

This is evidence for the "annotation standard completeness" + "quality
control mechanisms" scoring sub-items in the competition rubric.

Usage:
    uv run python scripts/annotator_agreement.py \\
        --annotator_a annotation/export_yudha.json \\
        --annotator_b annotation/export_budi.json \\
        --output annotation/agreement.json
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from jiwer import cer as jiwer_cer


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def extract_transcriptions(export_path: Path) -> dict[str, str]:
    """
    Read a Label Studio JSON export and return {image_basename: joined_transcription}.
    Joins multi-region transcriptions with newlines in top-to-bottom reading order.
    """
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError(
            f"{export_path}: expected a JSON array (export as 'JSON', not 'JSON-MIN')"
        )

    out: dict[str, str] = {}
    for task in tasks:
        image_raw = (task.get("data") or {}).get("image") or ""
        image_name = image_raw.rsplit("/", 1)[-1]
        # Strip Label Studio's upload hash prefix if present
        if "-" in image_name:
            first, rest = image_name.split("-", 1)
            if len(first) == 8 and all(c in "0123456789abcdef" for c in first.lower()):
                image_name = rest

        anns = task.get("annotations") or []
        if not anns:
            continue
        annotation = next((a for a in anns if not a.get("was_cancelled")), anns[0])

        # Collect region id → (y, transcription) so we can sort top-to-bottom
        regions: dict[str, dict] = {}
        for r in annotation.get("result", []):
            rid = r.get("id")
            if not rid:
                continue
            rtype = r.get("type")
            value = r.get("value", {})
            if rtype == "rectanglelabels":
                labels = value.get("rectanglelabels", [])
                regions.setdefault(rid, {})["label"] = labels[0] if labels else "line"
                regions[rid]["y"] = value.get("y", 0)
            elif rtype == "textarea":
                texts = value.get("text", [])
                regions.setdefault(rid, {})["text"] = " ".join(
                    t.strip() for t in texts if t
                ).strip()

        legible = [
            (r.get("y", 0), r.get("text", ""))
            for r in regions.values()
            if r.get("label", "line") != "illegible" and r.get("text")
        ]
        legible.sort(key=lambda p: p[0])
        joined = "\n".join(t for _, t in legible)
        if image_name and joined:
            out[image_name] = joined
    return out


def pairwise_cer(ref: str, hyp: str) -> float:
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return float(jiwer_cer(ref_n, hyp_n))


def main():
    ap = argparse.ArgumentParser(
        description="Inter-annotator agreement (pairwise CER) between two Label Studio exports.",
    )
    ap.add_argument("--annotator_a", type=Path, required=True, help="Annotator A's export JSON")
    ap.add_argument("--annotator_b", type=Path, required=True, help="Annotator B's export JSON")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("annotation/agreement.json"),
        help="Where to write the detailed per-image report (default: annotation/agreement.json)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print every non-zero-CER pair with both transcriptions",
    )
    args = ap.parse_args()

    a_path, b_path = args.annotator_a, args.annotator_b
    if not a_path.exists() or not b_path.exists():
        print(f"ERROR: one of the export files is missing", file=sys.stderr)
        sys.exit(1)

    trans_a = extract_transcriptions(a_path)
    trans_b = extract_transcriptions(b_path)

    shared = sorted(set(trans_a) & set(trans_b))
    if not shared:
        print("ERROR: no images annotated by both annotators — check the exports", file=sys.stderr)
        sys.exit(1)

    per_image = []
    total_cer = 0.0
    exact_matches = 0
    for image in shared:
        a_text = trans_a[image]
        b_text = trans_b[image]
        c = pairwise_cer(a_text, b_text)
        exact = normalize(a_text) == normalize(b_text)
        per_image.append({
            "image":       image,
            "annotator_a": a_text,
            "annotator_b": b_text,
            "cer":         round(c, 4),
            "exact":       exact,
        })
        total_cer += c
        exact_matches += int(exact)
        if args.verbose and not exact:
            print(f"  {image}  CER={c:.4f}")
            print(f"    A: {a_text}")
            print(f"    B: {b_text}")

    n = len(shared)
    mean = total_cer / n
    a_only = sorted(set(trans_a) - set(trans_b))
    b_only = sorted(set(trans_b) - set(trans_a))

    summary = {
        "shared_images":    n,
        "annotator_a_only": len(a_only),
        "annotator_b_only": len(b_only),
        "mean_cer":         round(mean, 4),
        "exact_matches":    exact_matches,
        "exact_rate":       round(exact_matches / n, 4),
        "per_image":        per_image,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 60)
    print(f"  Annotator A               : {a_path}  ({len(trans_a)} images)")
    print(f"  Annotator B               : {b_path}  ({len(trans_b)} images)")
    print(f"  Shared images             : {n}")
    print(f"  A-only / B-only           : {len(a_only)} / {len(b_only)}")
    print(f"  Mean pairwise CER         : {mean:.4f} ({mean*100:.2f}%)")
    print(f"  Exact-match rate          : {exact_matches}/{n} ({exact_matches/n*100:.2f}%)")
    print("=" * 60)
    print(f"Full report written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
