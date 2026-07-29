#!/usr/bin/env python3
"""Fine-tune TrOCR on historical Russian metrical book text."""
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

DATA_DIR = Path("finetune_data")

# Load dataset
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

train_data = load_jsonl(DATA_DIR / "train.jsonl")
val_data = load_jsonl(DATA_DIR / "val.jsonl")

print(f"Loaded {len(train_data)} train, {len(val_data)} val samples")

# Load processor and model
print("Loading TrOCR processor and model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Add Russian-specific characters to tokenizer
russian_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
russian_chars_upper = russian_chars.upper()
pre_1918_chars = "ѣѳіѵѯѱѡѫ"
all_new_chars = russian_chars + russian_chars_upper + pre_1918_chars

# Add special tokens
special_tokens = {"additional_special_tokens": list(all_new_chars)}
processor.tokenizer.add_special_tokens(special_tokens)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Configure
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Use MPS (Metal GPU) if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Dataset
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
        
        # Replace padding token id with -100 to ignore loss
        labels = [
            -100 if token == self.processor.tokenizer.pad_token_id else token 
            for token in labels
        ]
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

train_dataset = OCRDataset(train_data, processor)
val_dataset = OCRDataset(val_data, processor)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=64,
    save_total_limit=2,
    fp16=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
trainer.train()

# Save final model
model.save_pretrained("./trocr-finetuned/final")
processor.save_pretrained("./trocr-finetuned/final")
print("Model saved to ./trocr-finetuned/final")
