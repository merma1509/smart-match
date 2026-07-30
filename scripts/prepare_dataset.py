#!/usr/bin/env python3
"""MODULE 1: Prepare Dataset — preprocess + layout + crop regions for OCR.
Запуск: python3 scripts/prepare_dataset.py --max_samples 200000
Результат: finetune_data/ (уже preprocessed изображения)
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

# Импортируем наши сервисы!
from app.services.light_preprocess import light_preprocess
from app.services.region_preprocess import preprocess_region

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "finetune_data"
OUTPUT_PREPROC = ROOT / "finetune_data_preprocessed"


def scan_pairs(max_samples=None):
    """Scan data/images recursively, pair with data/texts by filename stem."""
    img_dir = ROOT / "data" / "images"
    txt_dir = ROOT / "data" / "texts"
    
    count = 0
    if not img_dir.exists():
        logger.warning(f"Directory {img_dir} not found")
        return
    
    for root, dirs, files in os.walk(img_dir):
        for fname in sorted(files):
            if not (fname.endswith(".jpg") or fname.endswith(".png")):
                continue
            stem = Path(fname).stem
            txt = txt_dir / f"{stem}.txt"
            if not txt.exists():
                continue
            text = txt.read_text(encoding="utf-8").strip()
            if not text or len(text) < 2:
                continue
            yield (str(Path(root) / fname), text)
            count += 1
            if max_samples and count >= max_samples:
                return


def preprocess_and_save(img_path: str, text: str, idx: int, output_dir: Path) -> dict:
    """Preprocess image, apply layout analysis, crop, and save preprocessed region.
    Returns dict for JSONL or None if failed.
    """
    try:
        # 1. Загружаем изображение
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
        
        # 2. Применяем light preprocessing (как в production!)
        preprocessed = light_preprocess(
            img_bgr,
            apply_resize=False,  # Не ресайзим — изображения уже маленькие
            apply_border_removal=False
        )
        
        # 3. Сохраняем preprocessed изображение
        stem = Path(img_path).stem
        out_filename = f"{stem}_preprocessed.png"
        out_path = output_dir / out_filename
        cv2.imwrite(str(out_path), preprocessed)
        
        return {
            "image_path": str(out_path),
            "text": text,
            "original_path": img_path
        }
        
    except Exception as e:
        logger.debug(f"Error processing {img_path}: {e}")
        return None


def prepare(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREPROC.mkdir(parents=True, exist_ok=True)
    
    # 1. Сканируем все пары
    logger.info(f"Scanning pairs (max={args.max_samples})...")
    pairs = list(scan_pairs(args.max_samples))
    logger.info(f"Total raw pairs: {len(pairs)}")
    
    if args.limit:
        random.shuffle(pairs)
        pairs = pairs[:args.limit]
        logger.info(f"Limited to {args.limit} pairs")
    
    # 2. Preprocess все изображения и сохраняем
    logger.info("Preprocessing and saving images...")
    valid = []
    for idx, (img_path, text) in enumerate(tqdm(pairs)):
        result = preprocess_and_save(img_path, text, idx, OUTPUT_PREPROC)
        if result:
            valid.append(result)
    
    logger.info(f"Valid preprocessed pairs: {len(valid)}")
    
    # 3. Shuffle and split
    random.shuffle(valid)
    n = len(valid)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    train = valid[n_test + n_val:]
    val = valid[n_test:n_test + n_val]
    test = valid[:n_test]
    
    # 4. Save JSONL (ссылаемся на preprocessed изображения)
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(data)} samples to {path}")
    
    # 5. Сохраняем метаданные
    meta = {
        "total": len(valid),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "preprocessed_dir": str(OUTPUT_PREPROC),
    }
    with open(OUTPUT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Done! Preprocessed images in {OUTPUT_PREPROC}")
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=50000, help="Max pairs to scan")
    p.add_argument("--limit", type=int, default=0, help="Limit after scan (0 = no limit)")
    prepare(p.parse_args())
