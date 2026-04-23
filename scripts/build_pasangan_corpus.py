"""
build_pasangan_corpus.py
Generates a synthetic pasangan-stress corpus for Aksara Jawa OCR fine-tuning.

The main corpus (assets/corpus_jawa.txt) under-covers pasangan (stacked
consonant forms encoded as C + U+A9C0 pangkon + C). Pasangan is the hardest
visual element of Aksara Jawa, so the model needs denser exposure to each
possible form. This script programmatically builds sentences where every word
contains at least one pasangan cluster — cycling through all 20 base
consonants as both the first and second member of each cluster.

Output: assets/corpus_jawa_pasangan.txt (one sentence per line), intended to
be fed to scripts/generate_aksara.py alongside the main corpus.

Usage:
    uv run python scripts/build_pasangan_corpus.py \\
        --output assets/corpus_jawa_pasangan.txt \\
        --lines 150 \\
        --seed 42
"""

import argparse
import random
from pathlib import Path


# The 20 base Aksara Jawa consonants (carakan), in the traditional Hanacaraka
# order. Codepoints U+A984 through U+A997.
CARAKAN = [
    "ꦲ",  # ha   U+A9B2  (or ꦲ U+A9B2 — modern form)
    "ꦤ",  # na   U+A9A4
    "ꦕ",  # ca   U+A995
    "ꦫ",  # ra   U+A9AB
    "ꦏ",  # ka   U+A98F
    "ꦢ",  # da   U+A9A2
    "ꦠ",  # ta   U+A9A0
    "ꦱ",  # sa   U+A9B1
    "ꦮ",  # wa   U+A9AE
    "ꦭ",  # la   U+A9AD
    "ꦥ",  # pa   U+A9A5
    "ꦣ",  # dha  U+A9A3
    "ꦗ",  # ja   U+A997
    "ꦪ",  # ya   U+A9AA
    "ꦚ",  # nya  U+A99A
    "ꦩ",  # ma   U+A9A9
    "ꦒ",  # ga   U+A992
    "ꦧ",  # ba   U+A9A7
    "ꦛ",  # tha  U+A99B
    "ꦔ",  # nga  U+A994
]

# Common sandhangan (vowel marks / diacritics) to mix in for realism
SANDHANGAN = [
    "",   # a (default, no mark)
    "ꦶ",  # i  wulu
    "ꦸ",  # u  suku
    "ꦼ",  # e  pepet
    "ꦺ",  # é  taling
    "ꦺꦴ", # o  taling-tarung
]

PANGKON = "꧀"  # U+A9C0 — the virama that triggers the pasangan stacking
PADA_LINGSA = "꧈"  # sentence-internal separator
PADA_LUNGSI = "꧉"  # end-of-sentence marker


def syllable(c: str) -> str:
    """Consonant + optional sandhangan = one syllable."""
    return c + random.choice(SANDHANGAN)


def word_with_pasangan(min_clusters: int = 1, max_clusters: int = 3) -> str:
    """
    Build a single word containing one or more pasangan clusters.
    Pattern: [syllable] + pasangan + [syllable] + pasangan + ... + [syllable]
    A pasangan cluster is C + PANGKON + C, so we alternate: syllable, C꧀, syllable, ...
    """
    n_clusters = random.randint(min_clusters, max_clusters)
    parts: list[str] = [syllable(random.choice(CARAKAN))]
    for _ in range(n_clusters):
        # The "dead" consonant before pangkon — no sandhangan because pangkon
        # kills the inherent vowel by definition.
        dead_c = random.choice(CARAKAN)
        parts.append(dead_c + PANGKON)
        # The stacked consonant that follows — may have sandhangan.
        parts.append(syllable(random.choice(CARAKAN)))
    return "".join(parts)


def sentence(min_words: int = 2, max_words: int = 5) -> str:
    """2–5 pasangan-rich words joined with spaces, punctuated."""
    n = random.randint(min_words, max_words)
    words = [word_with_pasangan() for _ in range(n)]
    sep = " " if random.random() < 0.4 else PADA_LINGSA
    end = PADA_LUNGSI if random.random() < 0.3 else ""
    return sep.join(words) + end


def structured_stress_lines() -> list[str]:
    """
    Systematic coverage — one line per (c1, c2) pair cycling through CARAKAN.
    20 × 20 = 400 possible pairs; we take a diagonal sample of ~20 lines that
    cover every consonant in both positions at least once.
    """
    out: list[str] = []
    for i, c1 in enumerate(CARAKAN):
        c2 = CARAKAN[(i + 1) % len(CARAKAN)]
        c3 = CARAKAN[(i * 3 + 5) % len(CARAKAN)]
        out.append(f"{syllable(c1)}{c2}{PANGKON}{syllable(c3)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Generate a pasangan-stress corpus for Aksara Jawa OCR training."
    )
    ap.add_argument("--output", type=Path, default=Path("assets/corpus_jawa_pasangan.txt"))
    ap.add_argument("--lines", type=int, default=150, help="Random sentences to generate (default: 150)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Structured + random. Dedupe while preserving order.
    lines: list[str] = []
    seen: set[str] = set()

    for ln in structured_stress_lines():
        if ln not in seen:
            lines.append(ln)
            seen.add(ln)

    attempts = 0
    while len(lines) < args.lines and attempts < args.lines * 5:
        ln = sentence()
        if ln not in seen:
            lines.append(ln)
            seen.add(ln)
        attempts += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Stats — verify every line has at least one pasangan cluster
    pas_counts = [ln.count(PANGKON) for ln in lines]
    avg = sum(pas_counts) / len(pas_counts)
    print(f"Wrote {len(lines)} lines to {args.output.resolve()}")
    print(f"  pasangan clusters per line: min={min(pas_counts)}  max={max(pas_counts)}  avg={avg:.2f}")
    print(f"  unique consonants used as stacked second: {len(set(ln[ln.index(PANGKON)+1] for ln in lines if PANGKON in ln))}")
    print(f"\nSample:")
    for ln in lines[:5]:
        print(f"  {ln}")


if __name__ == "__main__":
    main()
