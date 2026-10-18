#!/usr/bin/env python3
"""Extract structured data from receipt images using a local open-weight vision model (Ollama).

Usage:
    python receipt_extract.py receipt1.jpg receipt2.png
    python receipt_extract.py ./receipts_folder
    python receipt_extract.py ./receipts_folder --csv out.csv
    python receipt_extract.py ./receipts_folder --model gemma3:4b
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import ollama
from pydantic import BaseModel, Field

DEFAULT_MODEL = "qwen2.5vl:7b"  # swap for any Ollama vision model, e.g. gemma3:4b
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Receipt(BaseModel):
    vendor: str = Field(description="Store or company name")
    date: Optional[str] = Field(description="Purchase date as YYYY-MM-DD, null if not visible")
    total: Optional[float] = Field(description="Final total paid, numbers only")
    currency: Optional[str] = Field(description="3-letter code like USD, INR, EUR; null if unknown")
    category: Literal["software", "travel", "meals", "equipment", "office", "other"]


PROMPT = (
    "You are reading a receipt or invoice image. Extract the vendor, date (YYYY-MM-DD), "
    "final total paid, currency, and the best-fitting expense category. "
    "Use null for anything you cannot clearly read. Do not guess."
)


def collect_images(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
        else:
            print(f"Skipping {item}: not an image or folder", file=sys.stderr)
    return files


def extract(image_path: Path, model: str) -> dict:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [str(image_path)]}],
        format=Receipt.model_json_schema(),  # forces valid JSON matching the schema
        options={"temperature": 0},
    )
    return Receipt.model_validate_json(response["message"]["content"]).model_dump()


def main():
    parser = argparse.ArgumentParser(description="Extract receipt details with a local model.")
    parser.add_argument("inputs", nargs="+", help="Image files and/or folders")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--csv", help="Also write results to this CSV file")
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if not images:
        sys.exit("No images found.")

    results = []
    for img in images:
        try:
            data = {"file": img.name, **extract(img, args.model)}
        except Exception as e:  # keep going if one receipt fails
            data = {"file": img.name, "error": str(e)}
        results.append(data)
        print(json.dumps(data, ensure_ascii=False))

    if args.csv:
        fields = ["file", "vendor", "date", "total", "currency", "category", "error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
