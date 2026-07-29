#!/usr/bin/env python3
"""Fix the missing preprocessor config and evaluate model."""
import json, torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

MODEL_DIR = Path("/Users/mnijonshuti/smart-match/app/models/ocr/fine_tuned_trocr")

print("Files in model directory:")
for f in MODEL_DIR.iterdir():
    print(f"  {f.name}")

# The model was saved but preprocessor_config.json may be missing
# Let's fix it by loading processor from original model and saving it
print("\nLoading processor from original taiga75/ru-trocr-1700s...")
processor = TrOCRProcessor.from_pretrained("taiga75/ru-trocr-1700s")
processor.save_pretrained(str(MODEL_DIR))
print("Processor config saved!")

# Now load the full model
print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Evaluate on 20 validation samples
print("\n=== Evaluating on 20 validation samples ===\n")
with open("finetune_data/val.jsonl") as f:
    samples = [json.loads(line) for line in f.readlines()[:20]]

correct = 0
total = 0

for sample in tqdm(samples):
    image = Image.open(sample["image_path"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    gt = sample["text"].strip()
    match = predicted == gt
    
    status = "✓" if match else "✗"
    print(f"{status} GT: '{gt}'")
    print(f"   Pred: '{predicted}'")
    
    if match:
        correct += 1
    total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
