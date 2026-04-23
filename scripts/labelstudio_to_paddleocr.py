"""
labelstudio_to_paddleocr.py
Converts a Label Studio JSON export into the repo's two annotation formats:
  - Label.txt (PaddleOCR detection format with pixel bounding boxes)
  - ground_truth.jsonl (conversation format consumed by convert_format.py)

Label Studio export structure (JSON, not JSON-MIN):
    [
      {
        "data": {"image": "/data/upload/.../aksara_0001.jpg"},
        "annotations": [
          {
            "result": [
              {
                "id": "region_abc",
                "type": "rectanglelabels",
                "original_width": 800,
                "original_height": 200,
                "value": {
                  "x": 5.0, "y": 10.0,
                  "width": 90.0, "height": 60.0,
                  "rectanglelabels": ["line"]
                }
              },
              {
                "id": "region_abc",
                "type": "textarea",
                "value": {"text": ["ꦲꦤꦕꦫꦏ"]}
              }
            ]
          }
        ]
      }
    ]

Rectangle coords are percentages of original image size; we convert to pixels.
Regions labeled `illegible` are dropped from the VLM jsonl but kept in Label.txt
with `illegibility: true` so they can still train a detector.

Usage:
    uv run python scripts/labelstudio_to_paddleocr.py \
        --input annotation/export.json \
        --image_dir data/real/ \
        --label_out data/real/Label.txt \
        --jsonl_out data/real/ground_truth.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


USER_PROMPT = "Baca teks Aksara Jawa dalam gambar ini."


def pct_to_pixel_box(value: dict, orig_w: int, orig_h: int) -> list[list[int]]:
    """Convert Label Studio percentage rect to 4-point pixel polygon."""
    x = value["x"] * orig_w / 100.0
    y = value["y"] * orig_h / 100.0
    w = value["width"] * orig_w / 100.0
    h = value["height"] * orig_h / 100.0
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def extract_image_name(data_value: str) -> str:
    """
    Strip Label Studio's upload prefix to get just the filename.
    Handles /data/upload/X/Y-filename.jpg → filename.jpg
    and /data/local-files/?d=.../filename.jpg → filename.jpg
    """
    name = data_value.rsplit("/", 1)[-1]
    # Label Studio prepends a random hash like "a1b2c3d4-filename.jpg" on uploads
    if "-" in name and len(name.split("-", 1)[0]) in (8, 9):
        parts = name.split("-", 1)
        if len(parts[0]) == 8 and all(c in "0123456789abcdef" for c in parts[0].lower()):
            name = parts[1]
    return name


def parse_task(task: dict) -> tuple[str, list[dict]] | None:
    """
    Parse one Label Studio task into (image_name, [region_dicts]).
    Each region dict: {"bbox": [[x,y]*4], "transcription": str, "label": str, "script": str | None}
    """
    data = task.get("data", {})
    image_raw = data.get("image") or data.get("image_url") or ""
    if not image_raw:
        return None
    image_name = extract_image_name(image_raw)

    annotations = task.get("annotations", [])
    if not annotations:
        return image_name, []

    # Use the most recent non-cancelled annotation
    annotation = next(
        (a for a in annotations if not a.get("was_cancelled")),
        annotations[0],
    )
    results = annotation.get("result", [])

    # Collect rectangle regions by id, then merge transcription + choices by id
    regions: dict[str, dict] = {}
    for r in results:
        rid = r.get("id")
        if not rid:
            continue
        rtype = r.get("type")
        value = r.get("value", {})

        if rtype == "rectanglelabels":
            labels = value.get("rectanglelabels", [])
            label = labels[0] if labels else "line"
            orig_w = r.get("original_width") or 0
            orig_h = r.get("original_height") or 0
            if not (orig_w and orig_h):
                continue
            regions.setdefault(rid, {})
            regions[rid]["bbox"] = pct_to_pixel_box(value, orig_w, orig_h)
            regions[rid]["label"] = label

        elif rtype == "textarea":
            texts = value.get("text", [])
            text = " ".join(t.strip() for t in texts if t).strip()
            regions.setdefault(rid, {})
            regions[rid]["transcription"] = text

        elif rtype == "choices":
            choices = value.get("choices", [])
            regions.setdefault(rid, {})
            regions[rid]["script"] = choices[0] if choices else None

    # Only keep regions that have a bbox
    region_list = [
        {
            "bbox": r["bbox"],
            "transcription": r.get("transcription", "").strip(),
            "label": r.get("label", "line"),
            "script": r.get("script"),
        }
        for r in regions.values()
        if "bbox" in r
    ]

    # Sort top-to-bottom, then left-to-right, so multi-line joins preserve reading order
    region_list.sort(key=lambda r: (r["bbox"][0][1], r["bbox"][0][0]))
    return image_name, region_list


def build_label_txt_entry(image_name: str, regions: list[dict]) -> str:
    """PaddleOCR Label.txt line: <image>\\t<json-array-of-regions>"""
    items = []
    for r in regions:
        illegible = r["label"] == "illegible"
        items.append(
            {
                "transcription": r["transcription"] if not illegible else "",
                "points": r["bbox"],
                "label": "aksara_jawa",
                "illegibility": illegible,
            }
        )
    return f"{image_name}\t{json.dumps(items, ensure_ascii=False)}"


def build_jsonl_record(image_name: str, regions: list[dict]) -> dict | None:
    """
    Conversation-format record for ERNIEKit pipeline. Joins multi-line
    transcriptions with newlines. Skips images with no legible regions.

    If every legible region shares the same script_type (printed/handwritten/
    manuscript), that tag is attached to the record for per-tier evaluation.
    """
    legible = [r for r in regions if r["label"] != "illegible" and r["transcription"]]
    if not legible:
        return None
    transcription = "\n".join(r["transcription"] for r in legible)

    script_types = {r.get("script") for r in legible if r.get("script")}
    record = {
        "image": image_name,
        "conversations": [
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": transcription},
        ],
    }
    if len(script_types) == 1:
        record["script_type"] = next(iter(script_types))
    return record


def main():
    ap = argparse.ArgumentParser(
        description="Convert Label Studio JSON export to PaddleOCR Label.txt + ground_truth.jsonl."
    )
    ap.add_argument("--input", required=True, help="Label Studio JSON export file")
    ap.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing the annotated images (for existence check)",
    )
    ap.add_argument(
        "--label_out",
        required=True,
        help="Output path for PaddleOCR Label.txt",
    )
    ap.add_argument(
        "--jsonl_out",
        required=True,
        help="Output path for conversation-format ground_truth.jsonl",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any annotated image is missing from image_dir",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    image_dir = Path(args.image_dir)
    label_out = Path(args.label_out)
    jsonl_out = Path(args.jsonl_out)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"ERROR: image directory not found: {image_dir}")
        sys.exit(1)

    tasks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        print("ERROR: expected a JSON array at the top level (export as JSON, not JSON-MIN)")
        sys.exit(1)

    label_lines: list[str] = []
    jsonl_records: list[dict] = []
    missing_images: list[str] = []
    empty_tasks = 0
    total_regions = 0
    illegible_regions = 0

    for task in tasks:
        parsed = parse_task(task)
        if parsed is None:
            continue
        image_name, regions = parsed

        if not (image_dir / image_name).exists():
            missing_images.append(image_name)

        if not regions:
            empty_tasks += 1
            continue

        total_regions += len(regions)
        illegible_regions += sum(1 for r in regions if r["label"] == "illegible")

        label_lines.append(build_label_txt_entry(image_name, regions))

        record = build_jsonl_record(image_name, regions)
        if record is not None:
            jsonl_records.append(record)

    if missing_images:
        print(f"\nWARNING: {len(missing_images)} annotated images not found in {image_dir}:")
        for name in missing_images[:10]:
            print(f"  - {name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        if args.strict:
            print("Aborting due to --strict")
            sys.exit(1)

    label_out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    jsonl_out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in jsonl_records) + "\n",
        encoding="utf-8",
    )

    print(f"\nTasks processed        : {len(tasks)}")
    print(f"Images with annotations: {len(label_lines)}")
    print(f"Images with no regions : {empty_tasks}")
    print(f"Total regions          : {total_regions}")
    print(f"  legible              : {total_regions - illegible_regions}")
    print(f"  illegible            : {illegible_regions}")
    print(f"\nLabel.txt written      : {label_out.resolve()}")
    print(f"ground_truth.jsonl     : {jsonl_out.resolve()}  ({len(jsonl_records)} records)")


if __name__ == "__main__":
    main()
