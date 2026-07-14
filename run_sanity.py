import cv2
from pathlib import Path
from app.services.light_preprocess import light_preprocess
from app.services.layout import analyze_layout
import glob

# Pick 3 different images
test_images = glob.glob("data/01-0203-0745-000600/*.jpg")

for img_path in test_images: #[:3]:
    print(f"\n{'='*60}")
    print(f"Testing: {img_path}")
    
    raw = cv2.imread(img_path)
    if raw is None:
        print(f"  Cannot load")
        continue
    
    print(f"  Original size: {raw.shape}")
    
    # Stage 1
    clean = light_preprocess(raw)
    print(f"  After light_preprocess: {clean.shape}")
    
    # Stage 2
    result = analyze_layout(clean)
    
    # Summary
    s = result["metadata"]["summary"]
    print(f"  Table: {s.get('tables', '?')} table(s)")
    print(f"  Cells: {s.get('handwritten_cells', '?')} handwritten + {s.get('printed_cells', '?')} printed")
    print(f"  Stamps: {s.get('stamps', '?')}")
    print(f"  Marginal notes: {s.get('marginal_notes', '?')}")
    print(f"  Page numbers: {s.get('page_numbers', '?')}")
    print(f"  Signatures: {s.get('signatures', '?')}")
    print(f"  Record blocks: {s.get('record_blocks', '?')}")
    print(f"  Total elements: {result['metadata']['num_elements']}")