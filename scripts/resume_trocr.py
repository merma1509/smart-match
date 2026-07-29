#!/usr/bin/env python3
"""Resume TrOCR fine-tuning from saved checkpoint.
Usage: python scripts/resume_trocr.py --epochs 3 --batch_size 16
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "finetune_data"
CHECKPOINT_DIR = ROOT / "app" / "models" / "ocr" / "fine_tuned_trocr"
MODEL_NAME = "taiga75/ru-trocr-1700s"

def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

class OcrData(Dataset):
    def __init__(self, items, processor, max_len=128):
        self.items = items
        self.processor = processor
        self.max_len = max_len
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 64), 255)
        pix = self.processor(img, return_tensors="pt", padding=True).pixel_values[0]
        lbl = self.processor.tokenizer(
            item["text"], padding="max_length", max_length=self.max_len,
            truncation=True, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pix, "labels": lbl}

def train(args):
    # 1. Load data  
    logger.info(f"Loading data from {DATA_DIR}")
    train_items = load_jsonl(DATA_DIR / "train.jsonl")
    val_items = load_jsonl(DATA_DIR / "val.jsonl")
    if args.limit:
        train_items = train_items[:args.limit]
        val_items = val_items[:max(100, args.limit // 10)]
    logger.info(f"Train: {len(train_items)}, Val: {len(val_items)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load processor from BASE model (same as during initial training)
    logger.info(f"Loading processor from base model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    
    # 3. Load model WEIGHTS from checkpoint (weights only, not tokenizer)
    logger.info(f"Loading model weights from checkpoint: {CHECKPOINT_DIR}")
    model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT_DIR)).to(device)
    
    # 4. Ensure config matches base processor's tokenizer
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    # Freeze encoder for light mode
    if args.mode == "light":
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        logger.info(f"Light mode: encoder frozen, decoder={trainable/1e6:.1f}M params")

    # 5. DataLoaders
    pad_id = processor.tokenizer.pad_token_id
    def collate(batch):
        pix = torch.stack([b["pixel_values"] for b in batch])
        lbl = torch.stack([b["labels"] for b in batch])
        lbl[lbl == pad_id] = -100
        return {"pixel_values": pix, "labels": lbl}

    train_loader = DataLoader(
        OcrData(train_items, processor, args.max_length),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        OcrData(val_items, processor, args.max_length),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # 6. Optimizer & Scheduler
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    sched = get_scheduler("cosine", opt, int(0.1 * total_steps), total_steps)

    # 7. Train
    logger.info(f"Training: {args.epochs} more epochs, {total_steps} steps, lr={args.lr}")
    best_val = float("inf")
    
    # Load previous best_val
    metrics_path = CHECKPOINT_DIR / "training_metrics.json"
    if metrics_path.exists():
        try:
            prev = json.load(open(metrics_path))
            best_val = prev.get("best_val_loss", float("inf"))
            logger.info(f"Previous best val loss: {best_val}")
        except Exception:
            pass

    for epoch in range(args.epochs):
        # Train
        model.train()
        tloss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            pix = batch["pixel_values"].to(device)
            lbl = batch["labels"].to(device)
            loss = model(pixel_values=pix, labels=lbl).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()
            tloss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        logger.info(f"Epoch {epoch+1} train loss: {tloss/len(train_loader):.4f}")

        # Validate
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                pix = batch["pixel_values"].to(device)
                lbl = batch["labels"].to(device)
                vloss += model(pixel_values=pix, labels=lbl).loss.item()
        avg_vloss = vloss / len(val_loader)
        logger.info(f"Epoch {epoch+1} val loss:   {avg_vloss:.4f}")

        if avg_vloss < best_val:
            best_val = avg_vloss
            model.save_pretrained(str(CHECKPOINT_DIR))
            processor.save_pretrained(str(CHECKPOINT_DIR))
            # Fix processor_config.json after save (transformers known issue)
            proc_cfg = CHECKPOINT_DIR / "processor_config.json"
            if proc_cfg.exists():
                cfg = json.loads(proc_cfg.read_text())
                if isinstance(cfg.get("image_processor"), dict):
                    cfg["image_processor"] = cfg["image_processor"].get("image_processor_type", "ViTImageProcessor")
                    proc_cfg.write_text(json.dumps(cfg, indent=2))
            logger.info(f"Saved best model (val_loss={avg_vloss:.4f})")
        else:
            latest_dir = CHECKPOINT_DIR.parent / "fine_tuned_trocr_latest"
            latest_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(latest_dir))
            processor.save_pretrained(str(latest_dir))

    with open(CHECKPOINT_DIR / "training_metrics.json", "w") as f:
        json.dump({"best_val_loss": best_val, "epochs_trained": args.epochs}, f, indent=2)

    # Inference test
    logger.info("Inference test:")
    model.eval()
    test_img = Image.open(val_items[0]["image_path"]).convert("RGB")
    test_pix = processor(test_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        gen_ids = model.generate(test_pix, max_length=64, num_beams=4)
    pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    logger.info(f"  GT:   «{val_items[0]['text']}»")
    logger.info(f"  Pred: «{pred}»")
    logger.info(f"Done! Model at {CHECKPOINT_DIR}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["light", "full"], default="light")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    logger.info("="*60); logger.info("RESUMING TrOCR FINE-TUNE")
    logger.info(f"  Epochs: {args.epochs}, Mode: {args.mode}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info("="*60)
    train(args)
