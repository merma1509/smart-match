#!/usr/bin/env python3
"""Fine-tune TrOCR on 205k historical Russian images.

Usage:
    python scripts/fine_tune_trocr.py --max_samples 50000 --epochs 3

Запуск (в фоне):
    nohup python scripts/fine_tune_trocr.py --max_samples 30000 --epochs 3 > train.log 2>&1 &
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "taiga75/ru-trocr-1700s"
OUTPUT_DIR = ROOT / "app" / "models" / "ocr" / "fine_tuned_trocr"


class OcrDataset(Dataset):
    def __init__(self, pairs, processor, max_len=128):
        self.pairs = pairs
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 64), 255)
        pix = self.processor(img, return_tensors="pt", padding=True).pixel_values[0]
        lbl = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_len,
            truncation=True, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pix, "labels": lbl}


def load_pairs(max_samples=None):
    """Load (image_path, text) pairs sorted alphabetically."""
    d_img = ROOT / "data" / "images"
    d_txt = ROOT / "data" / "texts"
    pairs = []
    for fname in sorted(os.listdir(d_img)):
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue
        stem = Path(fname).stem
        txt = d_txt / f"{stem}.txt"
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        pairs.append((str(d_img / fname), text))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


def train(args):
    # Data
    logger.info(f"Loading up to {args.max_samples} pairs...")
    t0 = time.time()
    pairs = load_pairs(args.max_samples)
    logger.info(f"Loaded {len(pairs)} pairs in {time.time()-t0:.1f}s")
    random.shuffle(pairs)
    n_val = int(len(pairs) * 0.1)
    tr, va = pairs[n_val:], pairs[:n_val]
    logger.info(f"Train: {len(tr)}, Val: {len(va)}")

    # Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading {MODEL_NAME} on {device}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # DataLoaders
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        pix = torch.stack([b["pixel_values"] for b in batch])
        lbl = torch.stack([b["labels"] for b in batch])
        lbl[lbl == pad_id] = -100
        return {"pixel_values": pix, "labels": lbl}

    train_loader = DataLoader(
        OcrDataset(tr, processor, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        OcrDataset(va, processor, args.max_length),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0,
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = args.epochs * len(train_loader)
    sched = get_scheduler("cosine", opt, int(0.1 * total), total)

    # Train
    logger.info(f"Training: {args.epochs} epochs, {total} steps")
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tloss = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            pix = batch["pixel_values"].to(device)
            lbl = batch["labels"].to(device)
            loss = model(pixel_values=pix, labels=lbl).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tloss += loss.item()
        logger.info(f"Ep {epoch+1} train: {tloss/len(train_loader):.4f}")

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch+1}"):
                pix = batch["pixel_values"].to(device)
                lbl = batch["labels"].to(device)
                vloss += model(pixel_values=pix, labels=lbl).loss.item()
        avg_vloss = vloss / len(val_loader)
        logger.info(f"Ep {epoch+1} val: {avg_vloss:.4f}")

        if avg_vloss < best_val:
            best_val = avg_vloss
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(OUTPUT_DIR))
            processor.save_pretrained(str(OUTPUT_DIR))
            logger.info("Saved best model!")

    # Final save
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    json.dump({"best_val_loss": best_val, "num_samples": len(pairs)},
              open(OUTPUT_DIR / "metrics.json", "w"), indent=2)
    logger.info(f"Done! Model at {OUTPUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    train(p.parse_args())
