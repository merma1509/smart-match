# app/main.py
"""Smart Match API — Intelligent Information Extraction from Historical Metrical Books.
Innopolis University, 2026."""
import os
import uuid
import cv2
import json
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.light_preprocess import light_preprocess, _compute_quality_metrics
from app.services.layout import analyze_layout
from app.services.region_preprocess import preprocess_regions
from app.services.ocr import recognize_text, preload_models
from app.services.postprocessing import postprocess_ocr_text
from app.services.extraction import extract_information

app = FastAPI(
    title="Smart Match API",
    description="AI-powered document intelligence for historical Russian metrical books",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    logger.info("Starting Smart Match API...")
    preload_models()
    logger.info("Smart Match API ready")


@app.get("/")
def root():
    return {
        "service": "Smart Match API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /extract": "Extract genealogical data from image",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract structured genealogical data from a scanned metrical book page.
    
    Accepts: JPG, PNG images
    Returns: JSON with extracted fields and confidence scores
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # 1. Save uploaded file
    ext = Path(file.filename).suffix or ".jpg"
    input_path = UPLOAD_DIR / f"{request_id}{ext}"
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)
    
    logger.info(f"[{request_id}] Processing: {file.filename} ({len(content)} bytes)")
    
    try:
        # 2. Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise HTTPException(400, "Cannot read image file")
        
        logger.info(f"[{request_id}] Image: {image.shape}")
        
        # 3. Preprocess with quality metrics
        preprocessed, metrics = light_preprocess(image, max_dim=3000, return_metrics=True)
        
        # 4. Layout analysis with raw grayscale
        gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        if max(h, w) > 3000:
            scale = 3000 / max(h, w)
            gray_raw = cv2.resize(gray_raw, (int(w*scale), int(h*scale)))
        
        layout = analyze_layout(preprocessed, gray_raw=gray_raw)
        logger.info(f"[{request_id}] Layout: {layout['page_type']}, {len(layout['elements'])} elements")
        
        # 5. OCR on text regions
        texts = []
        for elem in layout["elements"]:
            if elem["type"] in ("data_cell", "text_block", "header_row", "record_block"):
                x1, y1, x2, y2 = elem["bbox"]
                crop = preprocessed[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                result = recognize_text(crop)
                if result["text"].strip():
                    texts.append(result["text"])
        
        full_text = "\n".join(texts)
        logger.info(f"[{request_id}] OCR text length: {len(full_text)} chars")
        
        # 6. Postprocess and extract
        postprocessed = postprocess_ocr_text(full_text)
        corrected_text = postprocessed["corrected_text"]
        extracted = extract_information(corrected_text)
        
        # 7. Build result
        elapsed = round(time.time() - start_time, 2)
        result = {
            "request_id": request_id,
            "file": file.filename,
            "processing_time_seconds": elapsed,
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "quality_metrics": metrics,
            "page_type": layout["page_type"],
            "num_elements": len(layout["elements"]),
            "extracted_data": extracted,
            "raw_text_preview": full_text[:500],
            "needs_review": extracted.get("needs_review", True),
        }
        
        # 8. Save result
        result_path = RESULTS_DIR / f"{request_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[{request_id}] Done in {elapsed}s | "
                   f"type={extracted.get('record_type', '?')} | "
                   f"review={extracted.get('needs_review', True)}")
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"[{request_id}] Failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")