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

from app.core.config import settings
from app.schemas.responses import BatchExtractionResponse, SingleExtractionResponse
from app.services.entity_resolution import resolve_entities
from app.services.extraction import extract_information
from app.services.layout import analyze_layout
from app.services.light_preprocess import light_preprocess
from app.services.ocr import recognize_text
from app.services.postprocessing import postprocess_ocr_text
from app.services.region_preprocess import preprocess_region

router = APIRouter()

UPLOAD_DIR = Path(settings.upload_dir)
RESULTS_DIR = Path(settings.output_dir)
ALLOWED_EXTENSIONS = {e for e in settings.allowed_extensions if e in {".jpg", ".jpeg", ".png"}}
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"File too large ({file.size} bytes). Max: {settings.max_file_size_mb}MB"
        )


def _determine_region_type(elem: dict) -> str:
    """Determine the preprocessing type for a layout element."""
    elem_type = elem["type"]
    props = elem.get("properties", {})

    # For data_cell, use the region_type property (handwritten vs printed)
    if elem_type == "data_cell":
        cell_type = props.get("region_type", "printed")
        if cell_type == "handwritten":
            return "handwritten"
        return "data_cell"

    # Map layout element types to preprocessing types
    type_map = {
        "text_block": "printed_text",
        "header_row": "printed_text",
        "record_block": "table_cell",
        "data_row": "table_cell",
    }
    return type_map.get(elem_type, "printed_text")


def _process_single_image(image_path: Path, file_name: str, request_id: str) -> dict:
    """Core processing logic reused by /extract and /extract/batch."""
    start_time = time.time()

    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(400, "Cannot read image file")

    logger.info(f"[{request_id}] Image: {image.shape}")

    # 1. Preprocess
    preprocessed, metrics = light_preprocess(
        image, max_dim=settings.max_image_dimension, return_metrics=True
    )

    # 2. Layout analysis
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    if max(h, w) > settings.max_image_dimension:
        scale = settings.max_image_dimension / max(h, w)
        gray_raw = cv2.resize(gray_raw, (int(w * scale), int(h * scale)))

    layout = analyze_layout(preprocessed, gray_raw=gray_raw)
    logger.info(f"[{request_id}] Layout: {layout['page_type']}, {len(layout['elements'])} elements")

    # 3. OCR on text regions with region-specific preprocessing
    texts = []
    for elem in layout["elements"]:
        if elem["type"] in ("data_cell", "text_block", "header_row", "record_block"):
            x1, y1, x2, y2 = elem["bbox"]
            crop = preprocessed[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Apply region-specific preprocessing before OCR
            region_type = _determine_region_type(elem)
            preprocessed_crop = preprocess_region(crop, region_type)

            result = recognize_text(preprocessed_crop)
            if result["text"].strip():
                texts.append(result["text"])

    full_text = "\n".join(texts)
    logger.info(f"[{request_id}] OCR text length: {len(full_text)} chars")

    # 4. Postprocess
    postprocessed = postprocess_ocr_text(full_text)
    corrected_text = postprocessed["corrected_text"]

    # 5. Extract information
    extracted = extract_information(corrected_text)

    # 6. Entity resolution (normalize names, resolve dates, compute age)
    resolved = resolve_entities(extracted)

    # 7. Build result — validate with Pydantic schema
    elapsed = round(time.time() - start_time, 2)
    result_dict = {
        "request_id": request_id,
        "file": file_name,
        "processing_time_seconds": elapsed,
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "quality_metrics": metrics,
        "page_type": layout["page_type"],
        "num_elements": len(layout["elements"]),
        "extracted_data": extracted,
        "resolved_data": resolved,
        "raw_text_preview": full_text[:500],
        "needs_review": extracted.get("needs_review", True),
    }

    # Validate against Pydantic schema
    SingleExtractionResponse(**result_dict)

    # 8. Save result
    result_path = RESULTS_DIR / f"{request_id}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{request_id}] Done in {elapsed}s | "
        f"type={extracted.get('record_type', '?')} | "
        f"review={extracted.get('needs_review', True)}"
    )
    return result_dict


@router.post("/extract", response_model=None)
async def extract(file: UploadFile = File(...)):
    """Extract structured genealogical data from a scanned metrical book page."""
    validate_file(file)
    request_id = str(uuid.uuid4())[:8]

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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


@router.post("/extract/batch", response_model=None)
async def extract_batch(files: list[UploadFile] = File(...)):
    """Extract from multiple images in one request.

    Транзакционная обработка: при сбое любого файла все ранее созданные
    результаты и загруженные файлы удаляются (rollback).
    """
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per batch request")

    results: list[dict] = []
    errors: list[dict] = []
    created_files: list[Path] = []
    created_results: list[Path] = []

    try:
        for file in files:
            file_request_id = str(uuid.uuid4())[:8]
            ext = os.path.splitext(file.filename or "")[1] or ".jpg"
            input_path = UPLOAD_DIR / f"{file_request_id}{ext}"

            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)
            created_files.append(input_path)

            result = _process_single_image(input_path, file.filename or "unknown", file_request_id)
            results.append(result)

            result_path = RESULTS_DIR / f"{file_request_id}.json"
            if result_path.exists():
                created_results.append(result_path)

    except Exception as e:
        logger.error(f"Batch processing failed, initiating rollback: {e}")
        # ROLLBACK: удаляем все созданные файлы и результаты
        for path in created_results:
            try:
                path.unlink(missing_ok=True)
                logger.info(f"Rollback: deleted result {path}")
            except Exception as cleanup_err:
                logger.warning(f"Rollback: failed to delete {path}: {cleanup_err}")

        for path in created_files:
            try:
                path.unlink(missing_ok=True)
                logger.info(f"Rollback: deleted upload {path}")
            except Exception as cleanup_err:
                logger.warning(f"Rollback: failed to delete {path}: {cleanup_err}")

        errors.append(
            {
                "file": getattr(file, "filename", "unknown"),
                "error": f"Batch aborted: {str(e)}",
            }
        )

        batch_response = BatchExtractionResponse(
            total=len(results) + len(errors),
            success=len(results),
            failed=len(errors),
            rollback=True,
            results=results,
            errors=errors,
        )
        return batch_response.model_dump()

    batch_response = BatchExtractionResponse(
        total=len(results) + len(errors),
        success=len(results),
        failed=len(errors),
        rollback=False,
        results=results,
        errors=errors,
    )
    return batch_response.model_dump()
