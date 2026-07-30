import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.core.config import settings

# Try EasyOCR
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. Run: pip install easyocr")

# Try fine-tuned TrOCR
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logger.debug("TrOCR unavailable (transformers/torch not installed). Using EasyOCR fallback.")

# ── Path to fine-tuned model ──
FINETUNED_TROCR_PATH = (
    Path(__file__).resolve().parent.parent.parent / "app" / "models" / "ocr" / "fine_tuned_trocr"
)


class OCREngine:
    _easyocr_reader = None
    _trocr_processor = None
    _trocr_model = None
    _trocr_device = None

    @classmethod
    def get_trocr(cls):
        """Lazy-load fine-tuned TrOCR model."""
        if cls._trocr_model is not None:
            return cls._trocr_processor, cls._trocr_model
        if not TROCR_AVAILABLE:
            return None, None
        model_path = FINETUNED_TROCR_PATH
        if not model_path.exists() or not (
            list(model_path.glob("*.safetensors")) or list(model_path.glob("*.bin"))
        ):
            logger.info(f"Fine-tuned model not found, using base: {settings.ocr_model_name}")
            model_path_str = settings.ocr_model_name
        else:
            model_path_str = str(model_path)
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Loading TrOCR from {model_path_str} on {device}...")
            cls._trocr_processor = TrOCRProcessor.from_pretrained(model_path_str)
            cls._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path_str).to(device)
            cls._trocr_model.eval()
            cls._trocr_device = device
            logger.info(
                f"TrOCR loaded ({sum(p.numel() for p in cls._trocr_model.parameters()) / 1e6:.0f}M params)"
            )
            return cls._trocr_processor, cls._trocr_model
        except Exception as e:
            logger.error(f"Failed to load TrOCR: {e}")
            return None, None

    @classmethod
    def get_easyocr(cls, use_gpu: bool = None):
        if cls._easyocr_reader is None and EASYOCR_AVAILABLE:
            gpu = use_gpu if use_gpu is not None else settings.ocr_use_gpu
            logger.info(f"Initializing EasyOCR with Russian... (gpu={gpu})")
            cls._easyocr_reader = easyocr.Reader(["ru"], gpu=gpu)
            logger.info("EasyOCR initialized")
        return cls._easyocr_reader

    def __init__(self, tesseract_lang: str = "rus"):
        self.tesseract_lang = tesseract_lang

    def recognize_trocr(self, image) -> dict:
        """Recognize text using fine-tuned TrOCR."""
        processor, model = self.get_trocr()
        if processor is None:
            return {"text": "", "confidence": 0.0, "model_used": "trocr_unavailable"}

        try:
            if isinstance(image, str):
                from PIL import Image as PILImage

                img = PILImage.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                from PIL import Image as PILImage

                img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img = image

            pixel_values = processor(img, return_tensors="pt").pixel_values.to(self._trocr_device)
            with torch.no_grad():
                gen = model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                )
            text = processor.batch_decode(gen, skip_special_tokens=True)[0]
            return {
                "text": text.strip(),
                "confidence": 0.8,  # TrOCR doesn't output confidence natively
                "model_used": "trocr",
            }
        except Exception as e:
            logger.error(f"TrOCR failed: {e}")
            return {"text": "", "confidence": 0.0, "model_used": "trocr_error"}

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
                "blocks": len(results),
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
                capture_output=True,
                text=True,
                timeout=30,
            )
            text = result.stdout.strip()
            return {"text": text, "confidence": 0.5 if text else 0.0, "model_used": "tesseract"}
        except Exception:
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
    """Recognize text with primary=TrOCR, secondary=EasyOCR, fallback=Tesseract."""
    engine = get_engine()

    # Primary: TrOCR (handwritten + old print)
    result = engine.recognize_trocr(image)
    if result["text"] and result["text"] != "":
        return result

    # Secondary: EasyOCR
    result = engine.recognize_easyocr(image)
    if result["confidence"] >= 0.3:
        return result

    # Fallback: Tesseract
    if use_voting:
        tess = engine.recognize_tesseract(image)
        if len(tess.get("text", "")) > len(result.get("text", "")):
            return tess

    return result


def preload_models():
    """Preload all OCR models on startup."""
    logger.info("Preloading OCR models...")
    # EasyOCR
    logger.info("Preloading EasyOCR...")
    OCREngine.get_easyocr()
    # TrOCR (lazy-loaded, but we trigger it)
    logger.info("Preloading TrOCR...")
    OCREngine.get_trocr()
    logger.info("All OCR models preloaded")


def cleanup_all():
    global _engine
    _engine = None
    OCREngine._easyocr_reader = None
    OCREngine._trocr_model = None
    OCREngine._trocr_processor = None
    logger.info("OCR resources freed")
