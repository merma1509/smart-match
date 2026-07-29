#!/usr/bin/env python3
"""Continue training the TrOCR model for more epochs."""
import json, torch
from pathlib import Path
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from torch.utils.data import Dataset
from loguru import logger

DATA_DIR = Path("finetune_data")
MODEL_DIR = Path("/Users/mnijonshuti/smart-match/app/models/ocr/fine_tuned_trocr")
OUTPUT_DIR = Path("./trocr-finetuned-v2")

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

train_data = load_jsonl(DATA_DIR / "train.jsonl")
val_data = load_jsonl(DATA_DIR / "val.jsonl")
logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val samples")

logger.info(f"Loading model from {MODEL_DIR}")
processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)

class OCRDataset(Dataset):
    def __init__(self, data, processor, max_length=64):
        self.data = data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_length, truncation=True
        ).input_ids
        labels = [
            -100 if token == self.processor.tokenizer.pad_token_id else token 
            for token in labels
        ]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

train_dataset = OCRDataset(train_data, processor)
val_dataset = OCRDataset(val_data, processor)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs-v2",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=64,
    save_total_limit=3,
    fp16=False,
    report_to="none",
    learning_rate=2e-5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

logger.info("Starting continued training (10 epochs)...")
trainer.train()

model.save_pretrained(str(OUTPUT_DIR / "final"))
processor.save_pretrained(str(OUTPUT_DIR / "final"))
logger.info(f"Model saved to {OUTPUT_DIR / 'final'}")
