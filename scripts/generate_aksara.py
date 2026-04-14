"""
generate_aksara.py
Synthetic Aksara Jawa image generator for PaddleOCR-VL fine-tuning.

Usage:
    uv run python scripts/generate_aksara.py --count 300 --output data/synthetic/
    uv run python scripts/generate_aksara.py --count 100 --output data/eval/ --eval
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_DIR  = PROJECT_DIR / "assets"
FONT_PATH   = ASSETS_DIR / "NotoSansJavanese-Regular.ttf"
CORPUS_PATH = ASSETS_DIR / "corpus_jawa.txt"

# ── Image dimensions ────────────────────────────────────────────────────────
IMG_WIDTHS  = [400, 512, 640, 800]
IMG_HEIGHTS = [64, 80, 96, 112, 128]

# ── Background colours (aged paper tones) ────────────────────────────────────
BACKGROUNDS = [
    (255, 255, 255),   # clean white
    (252, 248, 240),   # warm off-white
    (245, 240, 225),   # aged paper
    (240, 235, 215),   # old manuscript
    (250, 245, 235),   # cream
    (235, 230, 210),   # antique
]

# ── Text colours ─────────────────────────────────────────────────────────────
TEXT_COLORS = [
    (10,  10,  10),    # near black
    (30,  25,  20),    # dark brown-black
    (20,  20,  40),    # dark navy
    (40,  25,  10),    # dark sepia
    (15,  15,  15),    # soft black
]


def load_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")
    lines = [l.strip() for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    return lines


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Font not found: {FONT_PATH}\n"
                                f"Run: curl -L -o assets/NotoSansJavanese-Regular.ttf "
                                f"https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjavanese/NotoSansJavanese%5Bwght%5D.ttf")
    return ImageFont.truetype(str(FONT_PATH), size)


def render_image(
    text: str,
    font_size: int,
    img_w: int,
    img_h: int,
    bg_color: tuple,
    text_color: tuple,
) -> tuple[Image.Image, tuple]:
    """
    Render Aksara Jawa text onto a plain background.
    Returns (image, bounding_box) where bounding_box is (x1, y1, x2, y2).
    """
    img  = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    # Center with slight random offset
    max_x_off = max(0, img_w - tw - 20)
    max_y_off = max(0, img_h - th - 10)
    x = random.randint(10, max(10, max_x_off))
    y = random.randint(5,  max(5,  max_y_off))

    draw.text((x, y), text, font=font, fill=text_color)

    # Return tight bounding box of rendered text
    actual_bbox = (x, y, x + tw, y + th)
    return img, actual_bbox


def augment(img: Image.Image, severity: str = "medium") -> Image.Image:
    rng = random.random

    # Rotation
    max_angle = {"light": 2, "medium": 5, "heavy": 10}.get(severity, 5)
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) > 0.3:
        bg = img.getpixel((2, 2))
        img = img.rotate(angle, expand=False, fillcolor=bg)

    # Gaussian blur
    blur_prob = {"light": 0.2, "medium": 0.4, "heavy": 0.65}.get(severity, 0.4)
    if rng() < blur_prob:
        radius = random.uniform(0.3, {"light": 0.8, "medium": 1.5, "heavy": 2.5}[severity])
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Brightness / contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.30))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.25))

    # Noise
    noise_prob = {"light": 0.15, "medium": 0.45, "heavy": 0.75}.get(severity, 0.45)
    if rng() < noise_prob:
        arr = np.array(img).astype(np.int16)
        level = random.randint(3, {"light": 8, "medium": 15, "heavy": 25}[severity])
        noise = np.random.randint(-level, level, arr.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    # JPEG compression
    if rng() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(60, 95))
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


def to_label_txt(img_name: str, text: str, bbox: tuple) -> str:
    x1, y1, x2, y2 = bbox
    ann = [{
        "transcription": text,
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "label":  "aksara_jawa",
        "illegibility": False,
    }]
    return f"{img_name}\t{json.dumps(ann, ensure_ascii=False)}"


def to_ground_truth(img_name: str, text: str) -> str:
    return json.dumps({
        "image": img_name,
        "conversations": [
            {
                "role": "user",
                "content": "Baca teks Aksara Jawa dalam gambar ini.",
            },
            {
                "role": "assistant",
                "content": text,
            },
        ],
    }, ensure_ascii=False)


def generate(
    count:        int,
    output_dir:   str,
    seed:         int  = None,
    augmentation: str  = "mixed",
    preview:      bool = False,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    print(f"Loaded corpus: {len(corpus)} lines")

    # Severity distribution: 40% light / 40% medium / 20% heavy
    if augmentation == "mixed":
        pool = (["light"] * 40 + ["medium"] * 40 + ["heavy"] * 20) * (count // 100 + 2)
        random.shuffle(pool)
        sevs = pool[:count]
    else:
        sevs = [augmentation] * count

    labels, gts = [], []
    print(f"\nGenerating {count} images → {out.resolve()}\n")

    for i in range(count):
        name      = f"aksara_{str(i + 1).zfill(4)}.jpg"
        text      = random.choice(corpus)
        font_size = random.choice([18, 20, 22, 24, 26, 28, 32])
        img_w     = random.choice(IMG_WIDTHS)
        img_h     = random.choice(IMG_HEIGHTS)
        bg        = random.choice(BACKGROUNDS)
        fg        = random.choice(TEXT_COLORS)

        img, bbox = render_image(text, font_size, img_w, img_h, bg, fg)
        img       = augment(img, severity=sevs[i])
        img.save(out / name, format="JPEG", quality=random.randint(82, 95))

        labels.append(to_label_txt(name, text, bbox))
        gts.append(to_ground_truth(name, text))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:>4}/{count}]  {name}  ({sevs[i]})")

    # Save annotation files
    (out / "Label.txt").write_text(
        "\n".join(labels), encoding="utf-8")
    (out / "ground_truth.jsonl").write_text(
        "\n".join(gts), encoding="utf-8")
    (out / "dataset_stats.json").write_text(json.dumps({
        "total_images":  count,
        "corpus_lines":  len(corpus),
        "font":          str(FONT_PATH.name),
        "augmentation":  {s: sevs.count(s) for s in ("light", "medium", "heavy")},
        "seed":          seed,
        "task":          "aksara_jawa_ocr",
        "language":      "Javanese (Aksara Jawa)",
        "unicode_block": "U+A980–U+A9DF",
    }, indent=2, ensure_ascii=False))

    print(f"\n  Label.txt          → {out / 'Label.txt'}")
    print(f"  ground_truth.jsonl → {out / 'ground_truth.jsonl'}")
    print(f"  dataset_stats.json → {out / 'dataset_stats.json'}")
    print(f"\nDone. {count} images generated.")

    if preview:
        Image.open(out / "aksara_0001.jpg").show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Synthetic Aksara Jawa image generator.")
    ap.add_argument("--count",        type=int, default=300)
    ap.add_argument("--output",       type=str, default="data/synthetic/")
    ap.add_argument("--seed",         type=int, default=None)
    ap.add_argument("--augmentation", default="mixed",
                    choices=["light", "medium", "heavy", "mixed"])
    ap.add_argument("--eval",         action="store_true",
                    help="Eval mode: seed=42, light augmentation only")
    ap.add_argument("--preview",      action="store_true")
    args = ap.parse_args()

    if args.eval:
        generate(args.count, args.output,
                 seed=42, augmentation="light", preview=args.preview)
    else:
        generate(args.count, args.output,
                 seed=args.seed, augmentation=args.augmentation,
                 preview=args.preview)