from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import os
from app.services.region_preprocess import process_image, to_grayscale, clahe_contrast, limit_image_size
from app.services.layout_detection import detect_layout, segment_records
from app.services.ocr import OCREngine
from app.services.postprocessing import postprocess_ocr_text
from app.services.information_extraction import extract_information
from app.services.normalization import normalize_data
from app.services.validation import validate_data
from app.services.confidence_scoring import score_confidence
from app.services.entity_resolution import resolve_entities
from pydantic import BaseModel
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Initialize OCR engine once with Russian support
ocr_engine = None

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        ocr_engine = OCREngine(
            handwritten_model="taiga75/ru-trocr-1700s",
            printed_model="taiga75/ru-trocr-1700s",
            tesseract_lang="rus+eng"
        )
    return ocr_engine

def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large ({size} bytes). Max: {MAX_FILE_SIZE} bytes")

def preprocess_for_language(image, language: str):
    """Apply language-specific preprocessing.
    
    Russian documents: CLAHE enhancement only, no adaptive threshold
    English/German: Full preprocessing with adaptive threshold
    """
    if language == "ru":
        # Russian: gentle preprocessing — CLAHE + denoise only
        gray = to_grayscale(image)
        gray = limit_image_size(gray, 2000)
        gray = clahe_contrast(gray, clip_limit=2.0)
        return gray
    else:
        # Default: full preprocessing pipeline
        return process_image(
            image,
            max_image_dim=2000,
            apply_clahe=True,
            apply_deskew=True,
            apply_inpaint=False,
            apply_morphological=False,
            apply_crop=True,
            apply_border_removal=True,
            apply_resolution_norm=False
        )

@router.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    language: str = "auto"  # "auto", "ru", "en", "de"
):
    """
    AI-powered document intelligence endpoint.
    
    Processes scanned metrical book pages through a multi-stage pipeline:
    1. Image Preprocessing → 2. Layout Detection → 3. OCR → 
    4. Information Extraction → 5. Normalization & Validation → 
    6. Confidence Scoring → 7. Structured JSON Output
    
    Based on Smart Match system architecture [1].
    """
    # --- Step 0: Input Validation ---
    validate_file(file)
    contents = await file.read()
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    try:
        engine = get_ocr_engine()
        
        # Determine language settings
        if language == "auto":
            language = "ru"  # Default to Russian for historical docs
        
        # --- Step 1: Image Preprocessing ---
        logger.info("Step 1: Image preprocessing")
        processed = preprocess_for_language(image, language)
        
        # --- Step 2: Layout Detection ---
        logger.info("Step 2: Layout analysis and record detection")
        try:
            layout_result = detect_layout(processed)
            record_regions = segment_records(processed, layout_result)
            logger.info(f"Detected {len(record_regions)} record regions")
        except Exception as e:
            logger.warning(f"Layout detection failed, falling back to full image: {e}")
            record_regions = [processed]  # Fallback: process whole image
        
        # --- Step 3: OCR ---
        logger.info("Step 3: Optical Character Recognition")
        all_texts = []
        overall_confidence = 0.0
        
        for i, region in enumerate(record_regions):
            if language == "ru":
                ocr_result = engine.recognize_with_voting(
                    region, 
                    trocr_model="handwritten",
                    language_hint="ru"
                )
            elif language in ["en", "de"]:
                ocr_result = engine.recognize_with_voting(
                    region,
                    trocr_model="auto",
                    language_hint=language
                )
            else:
                ocr_result = engine.recognize_with_voting(region)
            
            all_texts.append(ocr_result["text"])
            overall_confidence += ocr_result.get("confidence", 0.0)
        
        raw_text = "\n".join(all_texts)
        overall_confidence = overall_confidence / len(record_regions) if record_regions else 0.0
        
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the image")
        
        # --- Step 4: Post-processing & Information Extraction ---
        logger.info("Step 4: Information extraction")
        postprocessed = postprocess_ocr_text(raw_text)
        clean_text = postprocessed["corrected_text"]
        
        extraction = extract_information(clean_text)
        record_type = extraction.get("record_type", "unknown")
        
        # --- Step 5: Data Normalization and Validation ---
        logger.info("Step 5: Data normalization and validation")
        normalized = normalize_data(extraction)
        validation_result = validate_data(normalized)
        
        # --- Step 6: Confidence Scoring ---
        logger.info("Step 6: Confidence scoring")
        scored = score_confidence(
            extraction_data=normalized,
            ocr_confidence=overall_confidence,
            postprocess_corrections=postprocessed["corrections_applied"],
            validation_errors=validation_result.get("errors", [])
        )
        
        # --- Step 7: Entity Resolution (optional enrichment) ---
        resolved = resolve_entities(scored)
        
        # --- Step 8: Build structured response ---
        response = {
            "filename": file.filename,
            "status": "processed",
            "language": language,
            "pipeline": {
                "preprocessing": "completed",
                "layout_detection": "completed" if len(record_regions) > 1 else "fallback_full_image",
                "ocr_engine": ocr_result.get("model_used", "unknown"),
                "information_extraction": "completed",
                "normalization": "completed",
                "validation": "completed",
                "confidence_scoring": "completed"
            },
            "record": {
                "record_type": record_type,
                "fields": resolved.get("fields", {}),
                "needs_review": validation_result.get("needs_review", False),
                "review_reasons": validation_result.get("review_reasons", [])
            },
            "confidence": {
                "overall": scored.get("overall_confidence", overall_confidence),
                "per_field": scored.get("field_confidences", {})
            },
            "metadata": {
                "ocr_confidence": overall_confidence,
                "corrections_applied": postprocessed["corrections_applied"],
                "validation_errors": validation_result.get("errors", []),
                "regions_detected": len(record_regions)
            }
        }
        
        logger.info(f"Successfully processed {file.filename}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")