"""
convert_format.py
Converts ground_truth.jsonl (conversation format) to ERNIEKit SFT format
required for PaddleOCR-VL fine-tuning with erniekit train.

ERNIEKit format:
    {
        "image_info": [{"matched_text_index": 0, "image_url": "./path/to/image.jpg"}],
        "text_info": [
            {"text": "OCR:", "tag": "mask"},
            {"text": "ꦲꦤꦏ꧀ꦏꦶ", "tag": "no_mask"}
        ]
    }

Usage:
    # Convert training set
    uv run python scripts/convert_format.py \
        --input data/synthetic/ground_truth.jsonl \
        --image_dir data/synthetic/ \
        --output training/ocr_vl_sft-train_aksara_jawa.jsonl

    # Convert eval set
    uv run python scripts/convert_format.py \
        --input data/eval/ground_truth.jsonl \
        --image_dir data/eval/ \
        --output training/ocr_vl_sft-test_aksara_jawa.jsonl

    # Merge train + real data into one file
    uv run python scripts/convert_format.py \
        --input data/synthetic/ground_truth.jsonl data/real/ground_truth.jsonl \
        --image_dir data/synthetic/ data/real/ \
        --output training/ocr_vl_sft-train_aksara_jawa.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def convert_record(record: dict, image_dir: Path, prompt: str = "OCR:") -> dict | None:
    """
    Convert one conversation-format record to ERNIEKit SFT format.

    Input:
        {
            "image": "aksara_0001.jpg",
            "conversations": [
                {"role": "user", "content": "Baca teks Aksara Jawa..."},
                {"role": "assistant", "content": "ꦲꦤꦏ꧀ꦏꦶ"}
            ]
        }

    Output:
        {
            "image_info": [{"matched_text_index": 0, "image_url": "./data/synthetic/aksara_0001.jpg"}],
            "text_info": [
                {"text": "OCR:", "tag": "mask"},
                {"text": "ꦲꦤꦏ꧀ꦏꦶ", "tag": "no_mask"}
            ]
        }
    """
    image_name = record.get("image", "")
    if not image_name:
        return None

    # Extract assistant response (the OCR transcription)
    transcription = ""
    for turn in record.get("conversations", []):
        if turn.get("role") == "assistant":
            transcription = turn.get("content", "").strip()
            break

    if not transcription:
        return None

    # Build the image URL relative path
    image_url = str(image_dir / image_name)

    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": transcription},
        ],
        "images": [image_url],
    }


def convert_file(
    input_path: Path,
    image_dir: Path,
    prompt: str = "OCR:",
) -> list[dict]:
    """Convert all records in a ground_truth.jsonl file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []
    skipped = 0

    for line_num, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  WARNING: skipping line {line_num} — JSON parse error: {e}")
            skipped += 1
            continue

        converted = convert_record(record, image_dir, prompt)
        if converted is None:
            print(f"  WARNING: skipping line {line_num} — missing image or transcription")
            skipped += 1
            continue

        records.append(converted)

    print(f"  Converted: {len(records)}  Skipped: {skipped}  Total: {len(records) + skipped}")
    return records


def main():
    ap = argparse.ArgumentParser(
        description="Convert ground_truth.jsonl to ERNIEKit PaddleOCR-VL SFT format."
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more ground_truth.jsonl files to convert",
    )
    ap.add_argument(
        "--image_dir",
        nargs="+",
        required=True,
        help="Image directory for each input file (must match --input count, or provide one for all)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="training/ocr_vl_sft-train_aksara_jawa.jsonl",
        help="Output ERNIEKit JSONL file path",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="OCR:",
        help="Task prompt prefix (default: 'OCR:')",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle records before writing (recommended when merging multiple sources)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    args = ap.parse_args()

    # Validate input/image_dir pairing
    inputs = [Path(p) for p in args.input]
    if len(args.image_dir) == 1:
        # Single image_dir applies to all inputs
        image_dirs = [Path(args.image_dir[0])] * len(inputs)
    elif len(args.image_dir) == len(inputs):
        image_dirs = [Path(p) for p in args.image_dir]
    else:
        print(
            f"ERROR: --image_dir must have 1 entry (shared) or {len(inputs)} entries (one per input)"
        )
        sys.exit(1)

    # Convert all input files
    all_records = []
    for input_path, image_dir in zip(inputs, image_dirs):
        print(f"\nConverting: {input_path}  (image_dir: {image_dir})")
        records = convert_file(input_path, image_dir, prompt=args.prompt)
        all_records.extend(records)

    if not all_records:
        print("ERROR: no records converted — check input files")
        sys.exit(1)

    # Optional shuffle
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(all_records)
        print(f"\nShuffled {len(all_records)} records (seed={args.seed})")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records),
        encoding="utf-8",
    )

    print(f"\nTotal records written : {len(all_records)}")
    print(f"Output                : {out_path.resolve()}")

    # Show one example
    print("\nSample record:")
    print(json.dumps(all_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()