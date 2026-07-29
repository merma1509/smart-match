#!/usr/bin/env python3
"""MODULE 1: Prepare Dataset — scan images, pair with texts, split train/val.
Запуск: python3 scripts/prepare_dataset.py --max_samples 30000
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "finetune_data"


def scan_pairs(max_samples=None):
    """Scan data/images + data/texts and return (img_path, text) list."""
    img_dir = ROOT / "data" / "images"
    txt_dir = ROOT / "data" / "texts"

    if not img_dir.exists():
        # Try subdirectories
        img_dir = ROOT / "data"
        for sub in sorted(os.listdir(img_dir)):
            sub_path = img_dir / sub
            if sub_path.is_dir() and sub.startswith("01-"):
                for fname in sorted(os.listdir(sub_path)):
                    if not (fname.endswith(".jpg") or fname.endswith(".png")):
                        continue
                    stem = Path(fname).stem
                    txt = txt_dir / f"{stem}.txt"
                    if not txt.exists():
                        continue
                    text = txt.read_text(encoding="utf-8").strip()
                    if not text or len(text) < 2:
                        continue
                    yield (str(sub_path / fname), text)
        return

    count = 0
    for fname in sorted(os.listdir(img_dir)):
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue
        stem = Path(fname).stem
        txt = txt_dir / f"{stem}.txt"
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text or len(text) < 2:
            continue
        yield (str(img_dir / fname), text)
        count += 1
        if max_samples and count >= max_samples:
            return


def prepare(args):
    """Main preparation function."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning pairs (max={args.max_samples})...")
    pairs = list(scan_pairs(args.max_samples))
    if args.max_samples and len(pairs) > args.max_samples:
        pairs = pairs[:args.max_samples]

    logger.info(f"Total pairs: {len(pairs)}")

    # Validate images
    valid = []
    for img_path, text in pairs:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w < 10 or h < 10:
                    continue
            valid.append({"image_path": img_path, "text": text})
        except Exception as e:
            logger.debug(f"Skipping {img_path}: {e}")

    logger.info(f"Valid pairs: {len(valid)}")

    # Shuffle and split
    random.shuffle(valid)
    n = len(valid)
    n_val = int(n * 0.1)
    train, val = valid[n_val:], valid[:n_val]

    # Save as JSONL
    for split_name, data in [("train", train), ("val", val)]:
        path = OUTPUT / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(data)} samples to {path}")

    # Stats
    text_lengths = [len(item["text"]) for item in valid]
    logger.info(f"Text length: min={min(text_lengths)}, avg={sum(text_lengths)/len(text_lengths):.1f}, max={max(text_lengths)}")

    # Save metadata
    meta = {
        "total": len(valid),
        "train": len(train),
        "val": len(val),
        "max_text_length": max(text_lengths),
        "avg_text_length": sum(text_lengths) / len(text_lengths),
    }
    with open(OUTPUT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Done! Metadata: {meta}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=50000,
                   help="Max image-text pairs to process")
    prepare(p.parse_args())
