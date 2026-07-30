#!/usr/bin/env python3
"""Prepare dataset with preprocessed images for OCR fine-tuning.
Использует layout detection + cropping для получения чистых регионов с текстом.
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

# Импортируем существующие сервисы
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.light_preprocess import light_preprocess
from app.services.layout import analyze_layout
from app.services.region_preprocess import preprocess_region

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "finetune_data_cropped"


def process_image(img_path: str, text: str, max_samples: int = None) -> list:
    """Process single image: preprocess → layout → crop → preprocess regions.
    Returns list of (cropped_image_path, text) pairs.
    """
    # 1. Load and preprocess
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        return []
    
    preprocessed = light_preprocess(image_bgr)
    
    # 2. Layout analysis
    layout = analyze_layout(preprocessed)
    
    # 3. Get relevant regions (handwritten cells, printed text blocks)
    results = []
    for elem in layout.get("elements", []):
        etype = elem["type"]
        bbox = elem["bbox"]
        
        # Нас интересуют ячейки с рукописным текстом
        if etype == "data_cell":
            props = elem.get("properties", {})
            if props.get("region_type") == "handwritten":
                x1, y1, x2, y2 = bbox
                crop = preprocessed[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    # Применяем region preprocessing
                    processed = preprocess_region(crop, "handwritten")
                    results.append((processed, text))
        
        # Или текстовые блоки
        elif etype in ("text_block", "printed_text"):
            x1, y1, x2, y2 = bbox
            crop = preprocessed[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                processed = preprocess_region(crop, "printed_text")
                results.append((processed, text))
    
    return results


def prepare(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    
    # Загружаем исходный датасет
    with open(ROOT / "finetune_data" / "train.jsonl") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    
    if args.max_samples:
        random.shuffle(train_data)
        train_data = train_data[:args.max_samples]
    
    logger.info(f"Processing {len(train_data)} samples...")
    
    new_pairs = []
    for idx, item in enumerate(tqdm(train_data)):
        img_path = item["image_path"]
        text = item["text"]
        
        try:
            results = process_image(img_path, text)
            for i, (processed_img, crop_text) in enumerate(results):
                # Сохраняем cropped изображение
                crop_filename = f"crop_{idx}_{i}.png"
                crop_path = OUTPUT_DIR / "images" / crop_filename
                cv2.imwrite(str(crop_path), processed_img)
                
                new_pairs.append({
                    "image_path": str(crop_path),
                    "text": crop_text,
                    "original": img_path
                })
        except Exception as e:
            logger.debug(f"Error processing {img_path}: {e}")
            continue
    
    # Split
    random.shuffle(new_pairs)
    n = len(new_pairs)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    
    train = new_pairs[n_test + n_val:]
    val = new_pairs[n_test:n_test + n_val]
    test = new_pairs[:n_test]
    
    # Save
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    meta = {
        "total": len(new_pairs),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Done! Created {len(new_pairs)} cropped pairs")
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=5000, help="Max samples to process")
    prepare(p.parse_args())
