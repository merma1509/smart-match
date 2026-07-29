#!/usr/bin/env python3
"""
MODULE 3: Evaluate OCR — compare fine-tuned model vs baseline on test set.
Запуск: python3 scripts/evaluate_ocr.py --samples 500

Output: app/models/ocr/fine_tuned_trocr/evaluation_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from jiwer import cer, wer
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "app" / "models" / "ocr" / "fine_tuned_trocr"
DATA_DIR = ROOT / "finetune_data"
BASELINE_MODEL = "taiga75/ru-trocr-1700s"


def load_jsonl(path, max_samples=None):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if max_samples and len(items) >= max_samples:
                break
    return items


def evaluate_model(model, processor, items, device, name="model"):
    """Evaluate CER and WER on a list of items."""
    model.eval()
    total_time = 0.0
    predictions = []
    references = []

    for item in tqdm(items, desc=f"Evaluating {name}"):
        img = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]

        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        total_time += time.time() - t0

        pred = processor.batch_decode(gen, skip_special_tokens=True)[0]
        predictions.append(pred)
        references.append(text)

    # Compute metrics
    cer_score = cer(references, predictions)
    wer_score = wer(references, predictions)
    avg_time = total_time / len(items)

    logger.info(f"{name}: CER={cer_score:.4f}, WER={wer_score:.4f}, "
                f"avg_time={avg_time:.3f}s")

    return {
        "name": name,
        "cer": cer_score,
        "wer": wer_score,
        "avg_inference_time": avg_time,
        "num_samples": len(items),
        "samples": [
            {"gt": ref, "pred": pred}
            for ref, pred in zip(references[:20], predictions[:20])
        ],
    }


def evaluate(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load test data
    val_items = load_jsonl(DATA_DIR / "val.jsonl", max_samples=args.samples)
    logger.info(f"Loaded {len(val_items)} evaluation samples")

    results = []

    # 1. Evaluate fine-tuned model
    ft_path = MODEL_PATH
    if (ft_path / "pytorch_model.bin").exists() or (ft_path / "model.safetensors").exists():
        logger.info("Loading fine-tuned model...")
        ft_processor = TrOCRProcessor.from_pretrained(str(ft_path))
        ft_model = VisionEncoderDecoderModel.from_pretrained(str(ft_path)).to(device)

        ft_results = evaluate_model(
            ft_model, ft_processor, val_items, device,
            name="fine_tuned_trocr",
        )
        results.append(ft_results)

    # 2. Evaluate baseline model
    logger.info("Loading baseline model...")
    base_processor = TrOCRProcessor.from_pretrained(BASELINE_MODEL)
    base_model = VisionEncoderDecoderModel.from_pretrained(BASELINE_MODEL).to(device)

    base_results = evaluate_model(
        base_model, base_processor, val_items, device,
        name="baseline_trocr",
    )
    results.append(base_results)

    # 3. Summary
    logger.info("=" * 50)
    logger.info("COMPARISON:")
    for r in results:
        logger.info(f"  {r['name']}: CER={r['cer']:.4f}, WER={r['wer']:.4f}")

    if len(results) == 2:
        cer_improvement = (results[1]["cer"] - results[0]["cer"]) / results[1]["cer"] * 100
        logger.info(f"  CER improvement: {cer_improvement:.1f}%")

    # Save results
    output_path = MODEL_PATH / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation to {output_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=500,
                   help="Number of validation samples to evaluate")
    evaluate(p.parse_args())
