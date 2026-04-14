"""
generate_aksara.py
Synthetic Aksara Jawa image generator for PaddleOCR-VL fine-tuning.

Supports two modes:
- Single-line: one sentence per image (text strip)
- Multi-line:  2-5 sentences stacked vertically (paragraph block)

Usage:
    uv run python scripts/generate_aksara.py --count 1000 --output data/synthetic/
    uv run python scripts/generate_aksara.py --count 150  --output data/eval/ --eval
    uv run python scripts/generate_aksara.py --count 1000 --output data/synthetic/ --erniekit --image_dir ./data/synthetic
    uv run python scripts/generate_aksara.py --count 500  --output data/synthetic/ --mode multiline
    uv run python scripts/generate_aksara.py --count 1000 --output data/synthetic/ --mode mixed
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

# ── Background colours (aged paper / clean tones) ───────────────────────────
BACKGROUNDS = [
    (255, 255, 255),
    (252, 248, 240),
    (245, 240, 225),
    (240, 235, 215),
    (250, 245, 235),
    (235, 230, 210),
    (248, 244, 230),
    (242, 237, 220),
]

# ── Text colours ─────────────────────────────────────────────────────────────
TEXT_COLORS = [
    (10,  10,  10),
    (30,  25,  20),
    (20,  20,  40),
    (40,  25,  10),
    (15,  15,  15),
    (25,  20,  30),
]

# ── Single-line image sizes ──────────────────────────────────────────────────
SINGLE_WIDTHS  = [400, 512, 640, 800]
SINGLE_HEIGHTS = [64, 80, 96, 112, 128]

# ── Multi-line image sizes ───────────────────────────────────────────────────
MULTI_WIDTHS  = [512, 640, 800, 960]
MULTI_HEIGHTS = [200, 256, 320, 400, 480]


def load_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")
    lines = [l.strip() for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not lines:
        raise ValueError("Corpus is empty")
    return lines


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Font not found: {FONT_PATH}\n"
            f"Run: curl -L -o assets/NotoSansJavanese-Regular.ttf "
            f"https://github.com/notofonts/javanese/raw/main/fonts/ttf/NotoSansJavanese-Regular.ttf"
        )
    return ImageFont.truetype(str(FONT_PATH), size)


# ── Single-line rendering ────────────────────────────────────────────────────

def render_single(
    text: str,
    font_size: int,
    img_w: int,
    img_h: int,
    bg_color: tuple,
    text_color: tuple,
) -> tuple[Image.Image, list[dict]]:
    """
    Render one line of Aksara Jawa text.
    Returns (image, annotations) where annotations is a list of one box.
    """
    img  = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    max_x = max(10, img_w - tw - 20)
    max_y = max(5,  img_h - th - 10)
    x = random.randint(10, max_x)
    y = random.randint(5,  max_y)

    draw.text((x, y), text, font=font, fill=text_color)

    annotations = [{
        "transcription": text,
        "points": [[x, y], [x + tw, y], [x + tw, y + th], [x, y + th]],
        "label": "aksara_jawa",
        "illegibility": False,
    }]
    return img, annotations


# ── Multi-line rendering ─────────────────────────────────────────────────────

def render_multiline(
    lines: list[str],
    font_size: int,
    img_w: int,
    img_h: int,
    bg_color: tuple,
    text_color: tuple,
    line_spacing_factor: float = 1.6,
) -> tuple[Image.Image, list[dict]]:
    """
    Render multiple lines of Aksara Jawa text as a paragraph block.
    Each line gets its own bounding box annotation.
    Returns (image, annotations).
    """
    img   = Image.new("RGB", (img_w, img_h), bg_color)
    draw  = ImageDraw.Draw(img)
    font  = load_font(font_size)
    anns  = []

    ref_bbox  = draw.textbbox((0, 0), "ꦲ", font=font)
    line_h    = ref_bbox[3] - ref_bbox[1]
    line_step = int(line_h * line_spacing_factor)

    margin_x = random.randint(16, 48)
    margin_y = random.randint(12, 32)
    y = margin_y

    for line_text in lines:
        if y + line_h > img_h - 10:
            break

        bbox = draw.textbbox((0, 0), line_text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]

        max_w = img_w - margin_x - 16
        if font.getlength(line_text) > max_w:
            while font.getlength(line_text) > max_w and len(line_text) > 2:
                line_text = line_text[:-1]
            bbox = draw.textbbox((0, 0), line_text, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]

        draw.text((margin_x, y), line_text, font=font, fill=text_color)

        anns.append({
            "transcription": line_text,
            "points": [
                [margin_x,      y],
                [margin_x + tw, y],
                [margin_x + tw, y + th],
                [margin_x,      y + th],
            ],
            "label": "aksara_jawa",
            "illegibility": False,
        })

        y += line_step

    return img, anns


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(img: Image.Image, severity: str = "medium") -> Image.Image:
    rng = random.random

    max_angle = {"light": 2, "medium": 5, "heavy": 10}.get(severity, 5)
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) > 0.3:
        bg = img.getpixel((2, 2))
        img = img.rotate(angle, expand=False, fillcolor=bg)

    blur_p = {"light": 0.2, "medium": 0.4, "heavy": 0.65}.get(severity, 0.4)
    if rng() < blur_p:
        radius = random.uniform(0.3, {"light": 0.8, "medium": 1.5, "heavy": 2.5}[severity])
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.30))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.25))

    noise_p = {"light": 0.15, "medium": 0.45, "heavy": 0.75}.get(severity, 0.45)
    if rng() < noise_p:
        arr   = np.array(img).astype(np.int16)
        level = random.randint(3, {"light": 8, "medium": 15, "heavy": 25}[severity])
        noise = np.random.randint(-level, level, arr.shape, dtype=np.int16)
        img   = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    if rng() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(60, 95))
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


# ── Annotation serialisation ─────────────────────────────────────────────────

def to_label_txt(img_name: str, annotations: list[dict]) -> str:
    return f"{img_name}\t{json.dumps(annotations, ensure_ascii=False)}"


def to_ground_truth(img_name: str, annotations: list[dict]) -> str:
    """Conversation format for reference and evaluation."""
    full_text = "\n".join(a["transcription"] for a in annotations)
    return json.dumps({
        "image": img_name,
        "conversations": [
            {
                "role": "user",
                "content": "Baca teks Aksara Jawa dalam gambar ini.",
            },
            {
                "role": "assistant",
                "content": full_text,
            },
        ],
    }, ensure_ascii=False)


def to_erniekit(img_name: str, annotations: list[dict], image_dir: str = "./data/synthetic") -> str:
    """ERNIEKit SFT format required for erniekit train command."""
    full_text = "\n".join(a["transcription"] for a in annotations)
    # Normalise path separator
    img_dir = image_dir.rstrip("/").rstrip("\\")
    return json.dumps({
        "image_info": [
            {
                "matched_text_index": 0,
                "image_url": f"{img_dir}/{img_name}",
            }
        ],
        "text_info": [
            {"text": "OCR:", "tag": "mask"},
            {"text": full_text, "tag": "no_mask"},
        ],
    }, ensure_ascii=False)


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(
    count:        int,
    output_dir:   str,
    seed:         int  = None,
    augmentation: str  = "mixed",
    mode:         str  = "mixed",
    erniekit:     bool = False,
    image_dir:    str  = None,
    preview:      bool = False,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    print(f"Loaded corpus: {len(corpus)} lines")

    # Severity distribution
    if augmentation == "mixed":
        pool = (["light"] * 40 + ["medium"] * 40 + ["heavy"] * 20) * (count // 100 + 2)
        random.shuffle(pool)
        sevs = pool[:count]
    else:
        sevs = [augmentation] * count

    # Mode distribution: 60% single-line, 40% multi-line
    if mode == "mixed":
        mode_pool = (["single"] * 60 + ["multiline"] * 40) * (count // 100 + 2)
        random.shuffle(mode_pool)
        modes = mode_pool[:count]
    else:
        modes = [mode] * count

    labels, gts, erniekit_records = [], [], []
    single_count, multi_count = 0, 0
    print(f"\nGenerating {count} images → {out.resolve()}\n")

    for i in range(count):
        name      = f"aksara_{str(i + 1).zfill(4)}.jpg"
        bg        = random.choice(BACKGROUNDS)
        fg        = random.choice(TEXT_COLORS)
        img_mode  = modes[i]
        sev       = sevs[i]

        if img_mode == "single":
            text      = random.choice(corpus)
            font_size = random.choice([18, 20, 22, 24, 26, 28, 32])
            img_w     = random.choice(SINGLE_WIDTHS)
            img_h     = random.choice(SINGLE_HEIGHTS)
            img, anns = render_single(text, font_size, img_w, img_h, bg, fg)
            single_count += 1
        else:
            n_lines   = random.randint(2, 5)
            lines     = random.choices(corpus, k=n_lines)
            font_size = random.choice([16, 18, 20, 22, 24])
            img_w     = random.choice(MULTI_WIDTHS)
            img_h     = random.choice(MULTI_HEIGHTS)
            img, anns = render_multiline(lines, font_size, img_w, img_h, bg, fg)
            multi_count += 1

        if not anns:
            anns = [{"transcription": "", "points": [[0,0],[1,0],[1,1],[0,1]],
                     "label": "aksara_jawa", "illegibility": True}]

        img = augment(img, severity=sev)
        img.save(out / name, format="JPEG", quality=random.randint(82, 95))

        labels.append(to_label_txt(name, anns))
        gts.append(to_ground_truth(name, anns))

        if erniekit:
            img_dir = image_dir or str(out)
            erniekit_records.append(to_erniekit(name, anns, img_dir))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:>4}/{count}]  {name}  ({img_mode}, {sev})")

    # Save annotation files
    (out / "Label.txt").write_text("\n".join(labels), encoding="utf-8")
    (out / "ground_truth.jsonl").write_text("\n".join(gts), encoding="utf-8")
    (out / "dataset_stats.json").write_text(json.dumps({
        "total_images":   count,
        "single_line":    single_count,
        "multi_line":     multi_count,
        "corpus_lines":   len(corpus),
        "font":           FONT_PATH.name,
        "augmentation":   {s: sevs.count(s) for s in ("light", "medium", "heavy")},
        "mode":           mode,
        "seed":           seed,
        "erniekit":       erniekit,
        "task":           "aksara_jawa_ocr",
        "language":       "Javanese (Aksara Jawa)",
        "unicode_block":  "U+A980–U+A9DF",
    }, indent=2, ensure_ascii=False))

    if erniekit:
        (out / "ocr_vl_sft.jsonl").write_text(
            "\n".join(erniekit_records), encoding="utf-8")
        print(f"  ocr_vl_sft.jsonl   → {out / 'ocr_vl_sft.jsonl'}")

    print(f"\n  Single-line images : {single_count}")
    print(f"  Multi-line images  : {multi_count}")
    print(f"  Label.txt          → {out / 'Label.txt'}")
    print(f"  ground_truth.jsonl → {out / 'ground_truth.jsonl'}")
    print(f"  dataset_stats.json → {out / 'dataset_stats.json'}")
    print(f"\nDone. {count} images generated.")

    if preview:
        Image.open(out / "aksara_0001.jpg").show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Synthetic Aksara Jawa image generator.")
    ap.add_argument("--count",        type=int, default=1000)
    ap.add_argument("--output",       type=str, default="data/synthetic/")
    ap.add_argument("--seed",         type=int, default=None)
    ap.add_argument("--augmentation", default="mixed",
                    choices=["light", "medium", "heavy", "mixed"])
    ap.add_argument("--mode",         default="mixed",
                    choices=["single", "multiline", "mixed"],
                    help="Image layout mode (default: mixed)")
    ap.add_argument("--erniekit",     action="store_true",
                    help="Also output ERNIEKit SFT format as ocr_vl_sft.jsonl")
    ap.add_argument("--image_dir",    type=str, default=None,
                    help="Image directory prefix used in ERNIEKit image_url paths")
    ap.add_argument("--eval",         action="store_true",
                    help="Eval preset: seed=42, light augmentation, single-line only")
    ap.add_argument("--preview",      action="store_true")
    args = ap.parse_args()

    if args.eval:
        generate(args.count, args.output,
                 seed=42, augmentation="light",
                 mode="single",
                 erniekit=args.erniekit,
                 image_dir=args.image_dir,
                 preview=args.preview)
    else:
        generate(args.count, args.output,
                 seed=args.seed, augmentation=args.augmentation,
                 mode=args.mode,
                 erniekit=args.erniekit,
                 image_dir=args.image_dir,
                 preview=args.preview)