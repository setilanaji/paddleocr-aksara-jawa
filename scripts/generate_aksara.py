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
FONTS_DIR   = ASSETS_DIR / "fonts"
# Backward-compat fallback for older repo layout (single font at assets/ root).
LEGACY_FONT_PATH = ASSETS_DIR / "NotoSansJavanese-Regular.ttf"
CORPUS_PATH          = ASSETS_DIR / "corpus_jawa.txt"
PASANGAN_CORPUS_PATH = ASSETS_DIR / "corpus_jawa_pasangan.txt"

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


def load_corpus(path: Path = CORPUS_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"Corpus is empty: {path}")
    return lines


def discover_fonts() -> list[Path]:
    """
    Find every .ttf file in assets/fonts/ — each is treated as a distinct
    typographic style for sampling during generation. Falls back to the legacy
    single-font location if the fonts folder is empty or missing.
    """
    fonts: list[Path] = []
    if FONTS_DIR.exists():
        fonts = sorted(FONTS_DIR.glob("*.ttf")) + sorted(FONTS_DIR.glob("*.TTF"))
    if not fonts and LEGACY_FONT_PATH.exists():
        fonts = [LEGACY_FONT_PATH]
    if not fonts:
        raise FileNotFoundError(
            f"No .ttf fonts found in {FONTS_DIR} or {LEGACY_FONT_PATH}.\n"
            f"Bootstrap with:\n"
            f"  curl -L -o assets/fonts/NotoSansJavanese-Regular.ttf \\\n"
            f"    https://github.com/notofonts/javanese/releases/download/NotoSansJavanese-v2.005/NotoSansJavanese-v2.005.zip"
        )
    return fonts


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def discover_real_backgrounds(real_bg_dir: Path | None) -> list[Path]:
    """Collect candidate real-manuscript images for semi-synthetic backgrounds."""
    if real_bg_dir is None:
        return []
    if not real_bg_dir.exists():
        print(f"  WARNING: --real_bg_dir {real_bg_dir} not found, skipping semi-synthetic mode")
        return []
    return sorted(real_bg_dir.glob("*.jpg")) + sorted(real_bg_dir.glob("*.jpeg")) \
         + sorted(real_bg_dir.glob("*.png"))


def build_background(
    img_w: int,
    img_h: int,
    bg_color: tuple,
    real_bg_path: Path | None = None,
) -> Image.Image:
    """
    Build the target-sized canvas the text will be drawn onto.

    If real_bg_path is None → solid background in bg_color (the classic path).
    Otherwise → semi-synthetic: crop a random patch from the real image,
    upscale if smaller than the target, heavy Gaussian blur to wash out any
    underlying script, and a small brightness jitter for variety. Ground
    truth remains the synthetic text that's drawn on top — the blurred
    underlying text is ignored noise.
    """
    if real_bg_path is None:
        return Image.new("RGB", (img_w, img_h), bg_color)

    with Image.open(real_bg_path) as src_file:
        src = src_file.convert("RGB")

    sw, sh = src.size
    # Ensure the source is at least 1.2× the target in both dims so the crop
    # doesn't cover the entire page (we want *patches*, not full pages).
    scale = max(img_w * 1.2 / sw, img_h * 1.2 / sh, 1.0) * random.uniform(1.0, 1.6)
    if scale > 1.0:
        src = src.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
        sw, sh = src.size

    x = random.randint(0, max(sw - img_w, 0))
    y = random.randint(0, max(sh - img_h, 0))
    patch = src.crop((x, y, x + img_w, y + img_h))

    # Heavy blur to remove legibility of the original script while keeping
    # paper texture + ink-tone gradients.
    patch = patch.filter(ImageFilter.GaussianBlur(radius=random.uniform(8.0, 16.0)))
    patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.85, 1.15))
    return patch


# ── Single-line rendering ────────────────────────────────────────────────────

def render_single(
    text: str,
    font_size: int,
    canvas: Image.Image,
    text_color: tuple,
    font_path: Path,
) -> tuple[Image.Image, list[dict]]:
    """
    Render one line of Aksara Jawa text onto the provided canvas.
    Returns (image, annotations) where annotations is a list of one box.
    """
    img  = canvas.copy()
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    font = load_font(font_path, font_size)

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
    canvas: Image.Image,
    text_color: tuple,
    font_path: Path,
    line_spacing_factor: float = 1.6,
) -> tuple[Image.Image, list[dict]]:
    """
    Render multiple lines of Aksara Jawa text as a paragraph block onto the
    provided canvas. Each line gets its own bounding box annotation.
    Returns (image, annotations).
    """
    img   = canvas.copy()
    img_w, img_h = img.size
    draw  = ImageDraw.Draw(img)
    font  = load_font(font_path, font_size)
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
            # Line too wide for the image — skip it entirely rather than
            # truncate mid-character (truncation would mismatch the GT text
            # against the rendered pixels and poison training).
            continue

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

def augment(
    img: Image.Image,
    severity: str = "medium",
    bg_color: tuple | None = None,
) -> Image.Image:
    rng = random.random

    max_angle = {"light": 2, "medium": 5, "heavy": 10}.get(severity, 5)
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) > 0.3:
        # Prefer the explicit background colour from the caller — safer than
        # sampling a pixel which can land on drawn text and bleed dark ink
        # into the rotated corners.
        fill = bg_color if bg_color is not None else img.getpixel((0, 0))
        img = img.rotate(angle, expand=False, fillcolor=fill)

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
    """
    PaddleFormers SFT messages format required by `paddleformers-cli train`
    with `train_dataset_type: messages` in the config. Upstream reference:
    https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/best_practices/PaddleOCR-VL
    """
    full_text = "\n".join(a["transcription"] for a in annotations)
    img_dir = image_dir.rstrip("/").rstrip("\\")
    return json.dumps({
        "messages": [
            {"role": "user", "content": "<image>OCR:"},
            {"role": "assistant", "content": full_text},
        ],
        "images": [f"{img_dir}/{img_name}"],
    }, ensure_ascii=False)


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(
    count:          int,
    output_dir:     str,
    seed:           int  = None,
    augmentation:   str  = "mixed",
    mode:           str  = "mixed",
    erniekit:       bool = False,
    image_dir:      str  = None,
    preview:        bool = False,
    pasangan_ratio: float = 0.0,
    real_bg_dir:    Path | None = None,
    real_bg_ratio:  float = 0.0,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded corpus: {len(corpus)} lines from {CORPUS_PATH.name}")

    pasangan_corpus: list[str] = []
    if pasangan_ratio > 0 and PASANGAN_CORPUS_PATH.exists():
        pasangan_corpus = load_corpus(PASANGAN_CORPUS_PATH)
        print(f"Loaded pasangan-stress corpus: {len(pasangan_corpus)} lines "
              f"(ratio {pasangan_ratio:.0%})")

    fonts = discover_fonts()
    print(f"Fonts available: {[f.name for f in fonts]}")

    real_backgrounds: list[Path] = []
    if real_bg_ratio > 0:
        real_backgrounds = discover_real_backgrounds(real_bg_dir)
        if real_backgrounds:
            print(f"Real-background pool: {len(real_backgrounds)} images from {real_bg_dir} "
                  f"(ratio {real_bg_ratio:.0%})")
        else:
            print(f"  WARNING: --real_bg_ratio={real_bg_ratio} but no backgrounds found — falling back to solid colours")
            real_bg_ratio = 0.0

    # Severity distribution — 40% light, 40% medium, 20% heavy when mixed.
    if augmentation == "mixed":
        sevs = random.choices(
            ["light", "medium", "heavy"], weights=[40, 40, 20], k=count
        )
    else:
        sevs = [augmentation] * count

    # Mode distribution — 60% single-line, 40% multi-line when mixed.
    if mode == "mixed":
        modes = random.choices(["single", "multiline"], weights=[60, 40], k=count)
    else:
        modes = [mode] * count

    labels, gts, erniekit_records = [], [], []
    single_count, multi_count = 0, 0
    print(f"\nGenerating {count} images → {out.resolve()}\n")

    for i in range(count):
        name     = f"aksara_{str(i + 1).zfill(4)}.jpg"
        img_mode = modes[i]
        sev      = sevs[i]

        # Pick the corpus for this image — pasangan-stress at the requested ratio.
        pick_pasangan = pasangan_corpus and random.random() < pasangan_ratio
        active_corpus = pasangan_corpus if pick_pasangan else corpus

        # Decide whether this image uses a real-manuscript background (semi-synthetic)
        # or a solid colour — keep the decision at the image level, not per-retry,
        # so we don't flip back and forth across attempts.
        use_real_bg = real_backgrounds and random.random() < real_bg_ratio

        # Retry with smaller font / different corpus lines until at least one
        # annotation fits. Avoids the old behaviour of emitting placeholder
        # illegibility:true boxes that would poison training.
        img, anns, bg = None, [], None
        for attempt in range(5):
            bg        = random.choice(BACKGROUNDS)
            fg        = random.choice(TEXT_COLORS)
            font_path = random.choice(fonts)
            real_bg_path = random.choice(real_backgrounds) if use_real_bg else None
            if img_mode == "single":
                text      = random.choice(active_corpus)
                font_size = random.choice([18, 20, 22, 24, 26, 28, 32])
                img_w     = random.choice(SINGLE_WIDTHS)
                img_h     = random.choice(SINGLE_HEIGHTS)
                canvas    = build_background(img_w, img_h, bg, real_bg_path)
                img, anns = render_single(text, font_size, canvas, fg, font_path)
            else:
                n_lines   = random.randint(2, 5)
                lines     = random.sample(active_corpus, k=min(n_lines, len(active_corpus)))
                # Start with the randomized font; shrink on retries so content fits.
                font_size = max(14, random.choice([16, 18, 20, 22, 24]) - attempt * 2)
                img_w     = random.choice(MULTI_WIDTHS)
                img_h     = random.choice(MULTI_HEIGHTS)
                canvas    = build_background(img_w, img_h, bg, real_bg_path)
                img, anns = render_multiline(lines, font_size, canvas, fg, font_path)
            if anns:
                break

        if not anns:
            # Extremely unusual — give up on this index and move on.
            print(f"  WARNING: could not render {name} after 5 attempts; skipping")
            continue

        if img_mode == "single":
            single_count += 1
        else:
            multi_count += 1

        img = augment(img, severity=sev, bg_color=bg)
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
        "total_images":     count,
        "single_line":      single_count,
        "multi_line":       multi_count,
        "corpus_lines":     len(corpus),
        "pasangan_corpus_lines": len(pasangan_corpus),
        "pasangan_ratio":   pasangan_ratio,
        "fonts":            [f.name for f in fonts],
        "real_bg_pool":     len(real_backgrounds),
        "real_bg_ratio":    real_bg_ratio,
        "augmentation":     {s: sevs.count(s) for s in ("light", "medium", "heavy")},
        "mode":             mode,
        "seed":             seed,
        "erniekit":         erniekit,
        "task":             "aksara_jawa_ocr",
        "language":         "Javanese (Aksara Jawa)",
        "unicode_block":    "U+A980–U+A9DF",
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
    ap.add_argument("--erniekit",       action="store_true",
                    help="Also output ERNIEKit SFT format as ocr_vl_sft.jsonl")
    ap.add_argument("--image_dir",      type=str, default=None,
                    help="Image directory prefix used in ERNIEKit image_url paths")
    ap.add_argument("--pasangan_ratio", type=float, default=0.0,
                    help="Fraction of images sampled from the pasangan-stress corpus "
                         "(assets/corpus_jawa_pasangan.txt); 0.0 disables, 0.2 recommended for training")
    ap.add_argument("--real_bg_dir",    type=Path, default=None,
                    help="Folder of real manuscript images to use as semi-synthetic backgrounds "
                         "(e.g. data/real/). Blurred + cropped to patches, text rendered on top")
    ap.add_argument("--real_bg_ratio",  type=float, default=0.0,
                    help="Fraction of images that use a real-background patch instead of solid colour")
    ap.add_argument("--eval",           action="store_true",
                    help="Eval preset: seed=42, light augmentation, single-line only, no pasangan stress")
    ap.add_argument("--preview",        action="store_true")
    args = ap.parse_args()

    if args.eval:
        generate(args.count, args.output,
                 seed=42, augmentation="light",
                 mode="single",
                 erniekit=args.erniekit,
                 image_dir=args.image_dir,
                 preview=args.preview,
                 pasangan_ratio=0.0,
                 real_bg_dir=None,
                 real_bg_ratio=0.0)
    else:
        generate(args.count, args.output,
                 seed=args.seed, augmentation=args.augmentation,
                 mode=args.mode,
                 erniekit=args.erniekit,
                 image_dir=args.image_dir,
                 preview=args.preview,
                 pasangan_ratio=args.pasangan_ratio,
                 real_bg_dir=args.real_bg_dir,
                 real_bg_ratio=args.real_bg_ratio)