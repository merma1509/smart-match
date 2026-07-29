#!/usr/bin/env python3
"""
Test fine-tuned TrOCR inference with correct processor loading.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "app" / "models" / "ocr" / "fine_tuned_trocr"
DATA_DIR = ROOT / "finetune_data"
MODEL_NAME = "taiga75/ru-trocr-1700s"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load processor from BASE (to avoid saved config corruption)
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
# Load model weights from fine-tuned checkpoint
model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))

# FIX: decoder_start_token_id must be set in BOTH configs
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.decoder.decoder_start_token_id = processor.tokenizer.cls_token_id
model.to(device)
model.eval()

print(f"Tokenizer: {type(processor.tokenizer).__name__}")
print(f"Vocab size: {processor.tokenizer.vocab_size}")
print(f"decoder_start_token_id (model.config): {model.config.decoder_start_token_id}")
print(f"decoder_start_token_id (model.config.decoder): {model.config.decoder.decoder_start_token_id}")

# Load val samples
val_path = DATA_DIR / "val.jsonl"
samples = []
with open(val_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

# Test on first 5 samples with different generation strategies
for strategy_name, gen_kwargs in [
    ("beams=4, rep_penalty=2.0", {
        "max_length": 64,
        "num_beams": 4,
        "repetition_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "do_sample": False,
    }),
    ("greedy, no penalty", {
        "max_length": 64,
        "num_beams": 1,
        "do_sample": False,
    }),
    ("beams=6, rep_penalty=1.5", {
        "max_length": 64,
        "num_beams": 6,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "do_sample": False,
    }),
]:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    for i in range(min(3, len(samples))):
        item = samples[i]
        img = Image.open(item["image_path"]).convert("RGB")
        pix = processor(img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
            gen = model.generate(pix, **gen_kwargs)
    pred = processor.batch_decode(gen, skip_special_tokens=True)[0]
    print(f"  Sample {i+1}: GT=«{item['text']}» → Pred=«{pred}»")

