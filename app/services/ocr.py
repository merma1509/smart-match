# app/services/ocr.py
"""OCR — Russian text recognition.
Primary: EasyOCR for Russian text
Fallback: Tesseract for clean printed text  
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
from loguru import logger
import subprocess
import tempfile

# Try EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. Run: pip install easyocr")


class OCREngine:
    _easyocr_reader = None

    @classmethod
    def get_easyocr(cls):
        if cls._easyocr_reader is None and EASYOCR_AVAILABLE:
            logger.info("Initializing EasyOCR with Russian...")
            cls._easyocr_reader = easyocr.Reader(['ru'], gpu=False)
            logger.info("EasyOCR initialized")
        return cls._easyocr_reader

    def __init__(self, tesseract_lang: str = "rus"):
        self.tesseract_lang = tesseract_lang

    def recognize_easyocr(self, image) -> dict:
        reader = self.get_easyocr()
        if reader is None:
            return self.recognize_tesseract(image)

        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)

        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        try:
            results = reader.readtext(img_rgb)
            if not results:
                return {"text": "", "confidence": 0.0, "model_used": "easyocr"}
            full_text = " ".join([text for _, text, _ in results])
            avg_confidence = sum([conf for _, _, conf in results]) / len(results)
            return {
                "text": full_text.strip(),
                "confidence": round(avg_confidence, 4),
                "model_used": "easyocr",
                "blocks": len(results)
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return self.recognize_tesseract(image)

    def recognize_tesseract(self, image, lang: str = None) -> dict:
        if lang is None:
            lang = self.tesseract_lang
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            return {"text": "", "confidence": 0.0, "model_used": "error"}

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, img)

        try:
            result = subprocess.run(
                ["tesseract", temp_path, "stdout", "-l", lang, "--psm", "6"],
                capture_output=True, text=True, timeout=30
            )
            text = result.stdout.strip()
            return {"text": text, "confidence": 0.5 if text else 0.0, "model_used": "tesseract"}
        except:
            return {"text": "", "confidence": 0.0, "model_used": "error"}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


_engine = None

def get_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine

def recognize_text(image, region_type: str = "printed", use_voting: bool = False) -> dict:
    engine = get_engine()
    result = engine.recognize_easyocr(image)
    if use_voting and result["confidence"] < 0.3:
        tess = engine.recognize_tesseract(image)
        if len(tess.get("text", "")) > len(result.get("text", "")):
            result = tess
    return result

def preload_models():
    logger.info("Preloading EasyOCR...")
    OCREngine.get_easyocr()
    logger.info("EasyOCR preloaded")

def cleanup_all():
    global _engine
    _engine = None
    OCREngine._easyocr_reader = None
    logger.info("OCR resources freed")