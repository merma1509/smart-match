"""diagnose_pipeline.py — Run full pipeline and show what's happening at each stage."""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from app.services.light_preprocess import light_preprocess
from app.services.layout import analyze_layout
from app.services.region_preprocess import preprocess_region

# Load a test image
img_path = "data/01-0203-0745-000600/00000003.jpg"
raw = cv2.imread(img_path)
if raw is None:
    print(f"❌ Cannot load {img_path}")
    exit(1)

print(f"Original image: {raw.shape}")

# Stage 1: Light preprocess
clean = light_preprocess(raw)
print(f"\n✅ Stage 1 — Light preprocess: {clean.shape}")

# Stage 2: Layout detection
layout = analyze_layout(clean)
print(f"\n✅ Stage 2 — Layout detection: {layout['page_type']} page")
print(f"   Total elements: {layout['metadata']['num_elements']}")
print(f"   Summary: {layout['metadata']['summary']}")

# Show all detected elements grouped by type
from collections import Counter
type_counts = Counter(e["type"] for e in layout["elements"])
print(f"\n📊 Elements by type:")
for t, count in type_counts.most_common():
    print(f"   {t}: {count}")

# Show first few elements with their bboxes
print(f"\n📦 First 10 elements:")
for i, elem in enumerate(layout["elements"][:10]):
    x1, y1, x2, y2 = elem["bbox"]
    w = x2 - x1
    h = y2 - y1
    print(f"   [{i}] {elem['type']:15s} ({x1:4d},{y1:4d},{x2:4d},{y2:4d}) [{w:4d}x{h:4d}] conf={elem['confidence']:.2f}")

# Stage 3: Region-specific preprocessing on first few cells
print(f"\n🔧 Stage 3 — Region preprocessing (first 5 data_cells):")
for i, elem in enumerate(layout["elements"]):
    if elem["type"] != "data_cell":
        continue
    if i >= 5:
        break
    
    x1, y1, x2, y2 = elem["bbox"]
    cropped = clean[y1:y2, x1:x2]
    cell_type = elem.get("properties", {}).get("region_type", "unknown")
    
    # Determine which preprocessor to use
    if cell_type == "handwritten":
        preproc_type = "handwritten"
    else:
        preproc_type = "printed_text"
    
    print(f"\n   Cell [{i}] type={cell_type}, size={cropped.shape}")
    print(f"      Cropped mean pixel: {np.mean(cropped):.1f}")
    print(f"      Cropped std: {np.std(cropped):.1f}")
    
    # Try preprocessing
    result = preprocess_region(cropped, preproc_type)
    if result is not None:
        print(f"      ✅ Preprocessed: shape={result.shape}, dtype={result.dtype}")
        print(f"      Post-process mean: {np.mean(result):.1f}")
        print(f"      Post-process unique values: {len(np.unique(result))}")
    else:
        print(f"      ❌ Preprocessing returned None!")
    
    # Save for visual inspection
    out_dir = Path("data/test_output")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / f"diagnose_cell_{i}_{cell_type}_original.jpg"), cropped)
    if result is not None:
        cv2.imwrite(str(out_dir / f"diagnose_cell_{i}_{cell_type}_processed.jpg"), result)
        print(f"      Saved visualizations to data/test_output/")