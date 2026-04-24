"""
extract_real_corpus.py
Extract Unicode Aksara Jawa line-level transcriptions from annotated real
manuscripts and write them as a corpus file for the synthetic generator.

Reads `data/real/ground_truth.jsonl` (produced by
scripts/labelstudio_to_paddleocr.py after a Label Studio export) and turns
each assistant-turn transcription into one or more corpus lines (split on
newlines — ground_truth.jsonl joins a page's lines with `\\n`).

Applies Unicode NFC normalisation to match the eval comparison (see
docs/data_report.md §8.1) and dedupes against the hand-crafted corpora so
the next synthetic regeneration can union them without repetition.

Output: `assets/corpus_jawa_real.txt` by default — a standalone file. To
make the generator use it, either:

  (a) cat assets/corpus_jawa_real.txt >> assets/corpus_jawa.txt
      (mutates the tracked corpus — reviewable via `git diff`)

  (b) adjust generate_aksara.py's CORPUS_PATH union logic to include this
      file (not done here; keeps this script single-purpose)

Usage:
    # Default: dry-run — print counts + sample
    uv run python scripts/extract_real_corpus.py

    # Write assets/corpus_jawa_real.txt
    uv run python scripts/extract_real_corpus.py --write

    # Custom input / output
    uv run python scripts/extract_real_corpus.py \\
        --input data/real/ground_truth.jsonl \\
        --output assets/corpus_jawa_real.txt \\
        --write
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path


AKSARA_JAWA_BLOCK = (0xA980, 0xA9DF)  # U+A980–U+A9DF
MIN_CHARS = 4  # drop very short fragments — one pada or a stray base consonant


def extract_lines(jsonl_path: Path) -> list[str]:
    """Pull assistant-turn transcriptions and split into per-line strings."""
    lines: list[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            convos = rec.get("conversations", [])
            for turn in convos:
                if turn.get("role") != "assistant":
                    continue
                content = turn.get("content", "")
                # Each ground_truth.jsonl record joins page lines with \n
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        lines.append(line)
    return lines


def has_aksara_jawa(s: str) -> bool:
    """True if the string contains at least one code point in the Aksara Jawa block."""
    lo, hi = AKSARA_JAWA_BLOCK
    return any(lo <= ord(c) <= hi for c in s)


def normalise(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_corpus(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        normalise(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=Path("data/real/ground_truth.jsonl"),
                    help="Annotated real-manuscript ground_truth.jsonl (default: data/real/ground_truth.jsonl)")
    ap.add_argument("--output", type=Path, default=Path("assets/corpus_jawa_real.txt"),
                    help="Output corpus file (default: assets/corpus_jawa_real.txt)")
    ap.add_argument("--existing", type=Path, nargs="+",
                    default=[Path("assets/corpus_jawa.txt"),
                             Path("assets/corpus_jawa_pasangan.txt")],
                    help="Hand-crafted corpus files to dedupe against")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS,
                    help=f"Drop lines shorter than this many chars (default: {MIN_CHARS})")
    ap.add_argument("--write", action="store_true",
                    help="Actually write the output file; otherwise just print the plan")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run scripts/labelstudio_to_paddleocr.py first.",
              file=sys.stderr)
        sys.exit(1)

    raw_lines = extract_lines(args.input)
    print(f"Read {len(raw_lines)} raw lines from {args.input}")

    # Clean + filter
    normed = (normalise(l) for l in raw_lines)
    filtered: list[str] = []
    dropped_short = dropped_no_aksara = 0
    for l in normed:
        if len(l) < args.min_chars:
            dropped_short += 1
            continue
        if not has_aksara_jawa(l):
            dropped_no_aksara += 1
            continue
        filtered.append(l)

    print(f"  dropped: {dropped_short} short (<{args.min_chars} chars), "
          f"{dropped_no_aksara} no-Aksara-Jawa-codepoint")

    # Dedupe against existing corpora + internal dedup
    existing: set[str] = set()
    for p in args.existing:
        n = len(load_corpus(p))
        existing |= load_corpus(p)
        print(f"  existing: {n} lines in {p}")

    seen: set[str] = set()
    unique_new: list[str] = []
    for l in filtered:
        if l in existing or l in seen:
            continue
        seen.add(l)
        unique_new.append(l)

    print(f"  new unique lines after dedup: {len(unique_new)}")
    if unique_new:
        print("  sample:")
        for l in unique_new[:5]:
            print(f"    {l}")

    if not args.write:
        print("\n--write not set; nothing written. Re-run with --write to emit",
              args.output)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(f"# Real-manuscript transcriptions extracted from {args.input}\n")
        f.write(f"# {len(unique_new)} lines, NFC-normalised, deduped vs "
                f"{', '.join(str(p) for p in args.existing)}\n")
        f.write("# Generated by scripts/extract_real_corpus.py — re-run after each\n"
                "# Label Studio export to refresh.\n")
        for l in unique_new:
            f.write(l + "\n")

    print(f"\nWrote {args.output} ({len(unique_new)} lines)")
    print("To use with generate_aksara.py, append to the main corpus:")
    print(f"  cat {args.output} >> assets/corpus_jawa.txt")
    print("Then regenerate synthetic/ and semi_synthetic/, push to R2, train v2.")


if __name__ == "__main__":
    main()
