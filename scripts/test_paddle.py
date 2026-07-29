"""Test PaddleOCR on the same samples for comparison."""
import json
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='russian', use_gpu=False)

with open("finetune_data/val.jsonl") as f:
    samples = [json.loads(line) for line in f.readlines()[:5]]

for i, sample in enumerate(samples):
    print(f"\n--- Sample {i+1} ---")
    print(f"GT: {sample['text']}")
    result = ocr.ocr(sample['image_path'], cls=True)
    text = ""
    if result and result[0]:
        text = " ".join([line[1][0] for line in result[0]])
    print(f"PaddleOCR: {text}")
