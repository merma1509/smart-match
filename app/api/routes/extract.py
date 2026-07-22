"""POST /extract endpoint — main pipeline."""
import json
import os
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.extraction import extract_information
from app.services.layout import analyze_layout
from app.services.light_preprocess import light_preprocess
from app.services.ocr import recognize_text
from app.services.postprocessing import postprocess_ocr_text

router = APIRouter()

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({file.size} bytes). Max: 50MB")


def _process_single_image(image_path: Path, file_name: str, request_id: str) -> dict:
    """Core processing logic reused by /extract and /extract/batch."""
    start_time = time.time()

    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(400, "Cannot read image file")

    logger.info(f"[{request_id}] Image: {image.shape}")

    # 1. Preprocess
    preprocessed, metrics = light_preprocess(image, max_dim=3000, return_metrics=True)

    # 2. Layout analysis
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    if max(h, w) > 3000:
        scale = 3000 / max(h, w)
        gray_raw = cv2.resize(gray_raw, (int(w * scale), int(h * scale)))

    layout = analyze_layout(preprocessed, gray_raw=gray_raw)
    logger.info(
        f"[{request_id}] Layout: {layout['page_type']}, {len(layout['elements'])} elements"
    )

    # 3. OCR on text regions
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

    # 4. Postprocess and extract
    postprocessed = postprocess_ocr_text(full_text)
    corrected_text = postprocessed["corrected_text"]
    extracted = extract_information(corrected_text)

    # 5. Build result
    elapsed = round(time.time() - start_time, 2)
    result = {
        "request_id": request_id,
        "file": file_name,
        "processing_time_seconds": elapsed,
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "quality_metrics": metrics,
        "page_type": layout["page_type"],
        "num_elements": len(layout["elements"]),
        "extracted_data": extracted,
        "raw_text_preview": full_text[:500],
        "needs_review": extracted.get("needs_review", True),
    }

    # 6. Save result
    result_path = RESULTS_DIR / f"{request_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{request_id}] Done in {elapsed}s | "
        f"type={extracted.get('record_type', '?')} | "
        f"review={extracted.get('needs_review', True)}"
    )
    return result


@router.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract structured genealogical data from a scanned metrical book page."""
    validate_file(file)
    request_id = str(uuid.uuid4())[:8]

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    input_path = UPLOAD_DIR / f"{request_id}{ext}"
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    logger.info(f"[{request_id}] Processing: {file.filename} ({len(content)} bytes)")

    try:
        result = _process_single_image(input_path, file.filename or "unknown", request_id)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")


@router.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...)):
    """Extract from multiple images in one request."""
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per batch request")

    results = []
    errors = []

    for file in files:
        file_request_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename or "")[1] or ".jpg"
        input_path = UPLOAD_DIR / f"{file_request_id}{ext}"

        try:
            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)

            result = _process_single_image(
                input_path, file.filename or "unknown", file_request_id
            )
            results.append(result)
        except Exception as e:
            logger.error(f"[{file_request_id}] Batch item failed: {e}")
            errors.append({
                "file": file.filename,
                "error": str(e),
            })

    return {
        "total": len(results) + len(errors),
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
