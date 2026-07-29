#!/usr/bin/env python3
"""Test the fine-tuned OCR model on a sample image."""
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load fine-tuned model
print("Loading fine-tuned model...")
processor = TrOCRProcessor.from_pretrained("./trocr-finetuned/final")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-finetuned/final")

# Test on a sample validation image
import json
with open("finetune_data/val.jsonl") as f:
    sample = json.loads(f.readline())

print(f"Image: {sample['image_path']}")
print(f"Expected text: {sample['text']}")

image = Image.open(sample["image_path"]).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Predicted text: {predicted}")
print(f"Match: {predicted.strip() == sample['text'].strip()}")
