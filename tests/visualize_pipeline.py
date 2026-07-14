#!/usr/bin/env python3
"""
Pipeline Visualization Script for Smart Match.
Shows: Original → Preprocessed → Layout Detection → OCR Regions → Extracted Data
"""
import sys
import os
from pathlib import Path
import json
import argparse

import cv2
import numpy as np
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.region_preprocess import process_image
from app.services.ocr import OCREngine
from app.services.extraction import InformationExtractor


def create_pipeline_visualization(
    image_path: str,
    output_dir: str = "pipeline_visualization",
    use_yolo: bool = False,
    max_dim: int = 2000,
    show_layout: bool = True,
    show_ocr: bool = True,
    save_all: bool = True,
) -> dict:
    """Run the full pipeline and create visualization images for each stage.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save visualizations
        use_yolo: Use YOLOv8 for layout detection (requires model file)
        max_dim: Max image dimension for preprocessing
        show_layout: Include layout detection stage
        show_ocr: Include OCR text overlay
        save_all: Save all intermediate images
        
    Returns:
        dict with paths to all generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    results = {}
    
    # STAGE 1: Original Image
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading original image")
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    orig_h, orig_w = original.shape[:2]
    logger.info(f"Original size: {orig_w}x{orig_h}")
    
    if save_all:
        orig_path = os.path.join(output_dir, f"{stem}_01_original.jpg")
        cv2.imwrite(orig_path, original)
        results["original"] = orig_path
        logger.info(f"  → Saved: {orig_path}")
    
    # STAGE 2: Preprocessed Image
    logger.info("-" * 60)
    logger.info("STAGE 2: Preprocessing")
    
    # FIXED: removed image_path and apply_denoise (not supported by your version)
    preprocessed = process_image(
        original,
        max_image_dim=max_dim,
        apply_clahe=True,
        apply_deskew=True,
        apply_border_removal=True,
        apply_crop=True,
        apply_threshold=False,
        # denoising is ALWAYS applied in your version - no flag needed
    )
    
    prep_h, prep_w = preprocessed.shape[:2]
    logger.info(f"Preprocessed size: {prep_w}x{prep_h}")
    
    if save_all:
        prep_path = os.path.join(output_dir, f"{stem}_02_preprocessed.jpg")
        cv2.imwrite(prep_path, preprocessed)
        results["preprocessed"] = prep_path
        logger.info(f"  → Saved: {prep_path}")
    
    # STAGE 3: Layout Detection (contour-based, no external models)
    if show_layout:
        logger.info("-" * 60)
        logger.info("STAGE 3: Layout Detection (contour-based)")
        
        # Convert grayscale back to BGR for visualization
        if len(preprocessed.shape) == 2:
            layout_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            layout_img = preprocessed.copy()
        
        try:
            # Use simple contour-based layout detection
            # (avoids dependency on LayoutDetector which may not exist)
            h, w = layout_img.shape[:2]
            
            # Simple column detection using vertical projection
            gray = preprocessed if len(preprocessed.shape) == 2 else cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Vertical projection
            vertical_proj = np.sum(binary == 255, axis=0) // 255
            
            # Find columns (gaps between text)
            col_threshold = h * 0.02  # 2% of height
            in_gap = True
            cols = []
            for x in range(w):
                if vertical_proj[x] < col_threshold:
                    if not in_gap:
                        cols.append((col_start, x))
                        in_gap = True
                else:
                    if in_gap:
                        col_start = x
                        in_gap = False
            
            # Horizontal projection for rows
            horizontal_proj = np.sum(binary == 255, axis=1) // 255
            row_threshold = w * 0.02
            in_gap = True
            rows = []
            for y in range(h):
                if horizontal_proj[y] < row_threshold:
                    if not in_gap:
                        rows.append((row_start, y))
                        in_gap = True
                else:
                    if in_gap:
                        row_start = y
                        in_gap = False
            
            # Create layout visualization
            layout_viz = layout_img.copy()
            
            # Draw detected rows (green)
            for y1, y2 in rows:
                cv2.rectangle(layout_viz, (0, y1), (w, y2), (0, 255, 0), 2)
            
            # Draw detected columns (blue)
            for x1, x2 in cols:
                cv2.rectangle(layout_viz, (x1, 0), (x2, h), (255, 0, 0), 2)
            
            # Draw record blocks (red) - approximate thirds
            third_h = h // 3
            for i in range(3):
                y1 = i * third_h
                y2 = (i + 1) * third_h
                cv2.rectangle(layout_viz, (10, y1 + 10), (w - 10, y2 - 10), (0, 0, 255), 3)
                cv2.putText(layout_viz, f"Record Block {i+1}", (20, y1 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw marginal note regions (cyan) - approximate left margin
            margin_w = w // 12
            cv2.rectangle(layout_viz, (0, 0), (margin_w, h), (255, 255, 0), 2)
            cv2.putText(layout_viz, "Marginal Notes", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Add summary text
            summary_text = f"Rows: {len(rows)}, Cols: {len(cols)}, Backend: contour"
            cv2.putText(layout_viz, summary_text, (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save layout visualization
            layout_path = os.path.join(output_dir, f"{stem}_03_layout.jpg")
            cv2.imwrite(layout_path, layout_viz)
            results["layout"] = layout_path
            logger.info(f"  → Saved: {layout_path}")
            
            # Save layout summary JSON
            layout_summary = {
                "backend": "contour",
                "table": {
                    "rows": len(rows),
                    "cols": len(cols),
                },
                "record_blocks": 3,
                "marginal_notes": 1,
                "orientation": "portrait" if h > w else "landscape",
            }
            summary_path = os.path.join(output_dir, f"{stem}_03_layout_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(layout_summary, f, indent=2, ensure_ascii=False)
            results["layout_summary"] = summary_path
            
        except Exception as e:
            logger.warning(f"Layout detection failed: {e}")
            layout_path = os.path.join(output_dir, f"{stem}_03_layout_failed.jpg")
            cv2.imwrite(layout_path, 
                       cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) 
                       if len(preprocessed.shape) == 2 else preprocessed)
            results["layout"] = layout_path
    
    # STAGE 4: OCR with Region Visualization
    if show_ocr:
        logger.info("-" * 60)
        logger.info("STAGE 4: OCR Recognition")
        
        try:
            ocr_engine = OCREngine.get_instance()
            
            # Create OCR visualization
            if len(preprocessed.shape) == 2:
                ocr_viz = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            else:
                ocr_viz = preprocessed.copy()
            
            # Split image into regions for OCR (top, middle, bottom thirds)
            h, w = ocr_viz.shape[:2]
            regions = [
                ("top", (0, 0, w, h // 3)),
                ("middle", (0, h // 3, w, h // 3 * 2)),
                ("bottom", (0, h // 3 * 2, w, h)),
            ]
            
            ocr_results = []
            for region_name, (rx, ry, rw, rh_bottom) in regions:
                region_img = preprocessed[ry:rh_bottom, rx:rw]
                if region_img.size == 0:
                    continue
                
                try:
                    result = ocr_engine.recognize(region_img)
                    text = result.get("text", "").strip()
                    conf = result.get("confidence", 0.0)
                    
                    if text:
                        ocr_results.append({
                            "region": region_name,
                            "text": text,
                            "confidence": conf,
                            "bbox": [rx, ry, rw, rh_bottom],
                        })
                        
                        # Draw region box
                        cv2.rectangle(ocr_viz, (rx, ry), (rw, rh_bottom), (255, 0, 255), 2)
                        
                        # Add label
                        label = f"[{region_name}] conf={conf:.2f}"
                        cv2.putText(ocr_viz, label, (rx + 5, ry + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                except Exception as e:
                    logger.warning(f"OCR failed for {region_name}: {e}")
            
            # Add OCR text overlay at the bottom
            ocr_texts = [r["text"][:80] for r in ocr_results if r["text"]]
            if ocr_texts:
                y_offset = h - 60
                for i, text in enumerate(ocr_texts):
                    cv2.putText(ocr_viz, f"OCR[{i}]: {text}", (10, y_offset + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Save OCR visualization
            ocr_path = os.path.join(output_dir, f"{stem}_04_ocr_regions.jpg")
            cv2.imwrite(ocr_path, ocr_viz)
            results["ocr_regions"] = ocr_path
            logger.info(f"  → Saved: {ocr_path}")
            
            # Save OCR results JSON
            ocr_json_path = os.path.join(output_dir, f"{stem}_04_ocr_results.json")
            with open(ocr_json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, indent=2, ensure_ascii=False)
            results["ocr_results"] = ocr_json_path
            
        except Exception as e:
            logger.warning(f"OCR visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    # STAGE 5: Information Extraction (output as JSON)
    logger.info("-" * 60)
    logger.info("STAGE 5: Information Extraction")
    
    try:
        extractor = InformationExtractor()
        
        # Use full OCR text
        full_text = ""
        if show_ocr and 'ocr_results' in dir():
            for r in ocr_results:
                full_text += r["text"] + "\n"
        
        if not full_text and show_ocr:
            # Fallback: OCR the whole image
            full_result = ocr_engine.recognize(preprocessed)
            full_text = full_result.get("text", "")
        
        if full_text:
            extraction_result = extractor.extract(full_text)
            
            # Save extraction result
            extract_path = os.path.join(output_dir, f"{stem}_05_extraction.json")
            with open(extract_path, "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            results["extraction"] = extract_path
            logger.info(f"  → Saved: {extract_path}")
            logger.info(f"  Record type: {extraction_result.get('record_type', 'unknown')}")
        else:
            logger.warning("No text extracted for information extraction")
            
    except Exception as e:
        logger.warning(f"Information extraction failed: {e}")
    
    # STAGE 6: Combined Pipeline Visualization
    logger.info("-" * 60)
    logger.info("STAGE 6: Creating combined pipeline visualization")
    
    try:
        # Create a combined image showing all stages
        viz_height = min(orig_h, 1200)
        
        # Scale factor
        scale = viz_height / orig_h
        
        # Stage 1: Original (resized)
        orig_small = cv2.resize(original, (int(orig_w * scale), viz_height))
        
        # Stage 2: Preprocessed
        if len(preprocessed.shape) == 2:
            prep_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            prep_bgr = preprocessed
        prep_small = cv2.resize(prep_bgr, (int(prep_w * scale), viz_height))
        
        # Layout
        if "layout" in results and os.path.exists(results["layout"]):
            layout_img_stage = cv2.imread(results["layout"])
            layout_h, layout_w = layout_img_stage.shape[:2]
            layout_small = cv2.resize(layout_img_stage, 
                                      (int(layout_w * viz_height / layout_h), viz_height))
        else:
            layout_small = np.ones((viz_height, viz_height // 2, 3), dtype=np.uint8) * 200
            cv2.putText(layout_small, "Layout: N/A", (10, viz_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # OCR
        if "ocr_regions" in results and os.path.exists(results["ocr_regions"]):
            ocr_img = cv2.imread(results["ocr_regions"])
            ocr_h, ocr_w = ocr_img.shape[:2]
            ocr_small = cv2.resize(ocr_img, 
                                   (int(ocr_w * viz_height / ocr_h), viz_height))
        else:
            ocr_small = np.ones((viz_height, viz_height // 2, 3), dtype=np.uint8) * 200
            cv2.putText(ocr_small, "OCR: N/A", (10, viz_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Get max widths for each row
        row1_w = max(orig_small.shape[1], prep_small.shape[1])
        row2_w = max(layout_small.shape[1], ocr_small.shape[1])
        
        # Pad images to same width
        def pad_to_width(img, target_w):
            h, w = img.shape[:2]
            if w < target_w:
                pad = np.ones((h, target_w - w, 3), dtype=np.uint8) * 255
                return cv2.hconcat([img, pad])
            return img
        
        orig_padded = pad_to_width(orig_small, row1_w)
        prep_padded = pad_to_width(prep_small, row1_w)
        layout_padded = pad_to_width(layout_small, row2_w)
        ocr_padded = pad_to_width(ocr_small, row2_w)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_padded, "ORIGINAL", (10, 30), font, 1.0, (0, 255, 0), 2)
        cv2.putText(prep_padded, "PREPROCESSED", (10, 30), font, 1.0, (0, 255, 0), 2)
        cv2.putText(layout_padded, "LAYOUT DETECTION", (10, 30), font, 1.0, (255, 0, 0), 2)
        cv2.putText(ocr_padded, "OCR REGIONS", (10, 30), font, 1.0, (255, 0, 0), 2)
        
        # Create rows
        gap = np.ones((5, max(row1_w, row2_w), 3), dtype=np.uint8) * 0
        sep = np.ones((viz_height, 5, 3), dtype=np.uint8) * 0
        
        row1 = cv2.hconcat([orig_padded, sep, prep_padded])
        row2 = cv2.hconcat([layout_padded, sep, ocr_padded])
        
        # Ensure same width
        final_w = max(row1.shape[1], row2.shape[1])
        if row1.shape[1] < final_w:
            pad = np.ones((row1.shape[0], final_w - row1.shape[1], 3), dtype=np.uint8) * 255
            row1 = cv2.hconcat([row1, pad])
        if row2.shape[1] < final_w:
            pad = np.ones((row2.shape[0], final_w - row2.shape[1], 3), dtype=np.uint8) * 255
            row2 = cv2.hconcat([row2, pad])
        
        # Stack vertically
        pipeline_viz = cv2.vconcat([row1, np.ones((5, final_w, 3), dtype=np.uint8) * 0, row2])
        
        pipeline_path = os.path.join(output_dir, f"{stem}_pipeline_complete.jpg")
        cv2.imwrite(pipeline_path, pipeline_viz)
        results["pipeline_complete"] = pipeline_path
        logger.info(f"  → Complete pipeline: {pipeline_path}")
        
    except Exception as e:
        logger.error(f"Pipeline visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("Pipeline visualization complete!")
    logger.info(f"Results saved to: {output_dir}/")
    
    return results


def display_results(results: dict):
    """Print a summary of generated files."""
    print("\n" + "=" * 60)
    print("PIPELINE VISUALIZATION RESULTS")
    print("=" * 60)
    
    stages = [
        ("Original Image", "original"),
        ("Preprocessed", "preprocessed"),
        ("Layout Detection", "layout"),
        ("Layout Summary", "layout_summary"),
        ("OCR Regions", "ocr_regions"),
        ("OCR Text", "ocr_results"),
        ("Extracted Data", "extraction"),
        ("Complete Pipeline", "pipeline_complete"),
    ]
    
    for label, key in stages:
        if key in results:
            print(f"  {label}: {results[key]}")
        else:
            print(f"  {label}: Not available")
    
    print("=" * 60)
    
    # Print extraction summary if available
    if "extraction" in results:
        with open(results["extraction"], "r", encoding="utf-8") as f:
            data = json.load(f)
        print("\nEXTRACTION SUMMARY:")
        print(f"  Record type: {data.get('record_type', 'N/A')}")
        for key, value in data.items():
            if key != "_extraction" and isinstance(value, dict) and "value" in value:
                print(f"  {key}: {value['value']} (confidence: {value.get('confidence', 'N/A')})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Match Pipeline Visualization"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", "-o", default="pipeline_visualization",
                        help="Output directory")
    parser.add_argument("--max-dim", type=int, default=2000,
                        help="Max image dimension")
    parser.add_argument("--no-layout", action="store_true",
                        help="Skip layout detection")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR visualization")
    parser.add_argument("--use-yolo", action="store_true",
                        help="Use YOLOv8 for layout (requires model)")
    parser.add_argument("--save-all", action="store_true", default=True,
                        help="Save all intermediate images")
    
    args = parser.parse_args()
    
    results = create_pipeline_visualization(
        args.image,
        output_dir=args.output_dir,
        use_yolo=args.use_yolo,
        max_dim=args.max_dim,
        show_layout=not args.no_layout,
        show_ocr=not args.no_ocr,
        save_all=args.save_all,
    )
    
    display_results(results)