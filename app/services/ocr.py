# Business logic: OCR — Russian historical text recognition
# Primary models:
#   - kazars24/trocr-base-handwritten-ru for handwriting (CER 4.85%)
#   - taiga75/ru-trocr-1700s for printed historical text (CER 1.69%)
# Fallback: Tesseract with rus for clean printed text
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from loguru import logger
import subprocess
import tempfile
import os
import hashlib
import json

# ── Model identifiers ──
MODEL_HANDWRITTEN = "kazars24/trocr-base-handwritten-ru"
MODEL_PRINTED_HISTORICAL = "taiga75/ru-trocr-1700s"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Отключаем токенизатор параллелизм

class OCREngine:
    """OCR engine for Russian historical texts.

    Dual-model architecture:
      - Handwritten regions  -> kazars24/trocr-base-handwritten-ru (CER 4.85%)
      - Printed regions      -> taiga75/ru-trocr-1700s (CER 1.69%)
      - Fallback             -> Tesseract (rus)

    Singleton pattern per model for memory efficiency.
    """

    _instances = {}  # model_name -> OCREngine

    @classmethod
    def get_instance(cls, model_name: str = MODEL_HANDWRITTEN, **kwargs):
        """Get or create a singleton OCR engine for a given model."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name=model_name, **kwargs)
        return cls._instances[model_name]

    def __init__(self,
                 model_name: str = MODEL_HANDWRITTEN,
                 tesseract_lang: str = "rus",
                 device: str = None,
                 use_gpu: bool = True):
        self.model_name = model_name
        self.tesseract_lang = tesseract_lang

        # Auto-detect device
        if device is None:
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            elif use_gpu and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"[{model_name}] Using device: {self.device}")

        # Load TrOCR model
        logger.info(f"[{model_name}] Loading model...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"[{model_name}] Model loaded successfully")

    # ── Image preparation ──
    def _prepare_image(self, image) -> Image.Image:
        """Convert various image formats to PIL RGB."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    # ── Cache ──
    def _get_cache_key(self, image) -> str:
        pil_image = self._prepare_image(image)
        return hashlib.md5(pil_image.tobytes()).hexdigest()

    def _check_cache(self, cache_key: str, cache_dir: str = "data/cache") -> dict | None:
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, result: dict, cache_dir: str = "data/cache"):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        with open(cache_path, "w") as f:
            json.dump({"text": result["text"], "confidence": result["confidence"]}, f, indent=2)

    def clear_cache(self, cache_dir: str = "data/cache"):
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(cache_dir, filename))

    # ── Recognition ──
    def recognize(self, image, return_confidence: bool = True, use_cache: bool = True) -> dict:
        """Recognize Russian text from an image using the loaded TrOCR model."""
        pil_image = self._prepare_image(image)

        # Cache check
        if use_cache:
            cache_key = self._get_cache_key(pil_image)
            cached = self._check_cache(cache_key)
            if cached:
                return cached

        # Inference
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Confidence
        if return_confidence:
            confidence = self._compute_confidence(pixel_values, generated_ids)
        else:
            confidence = 0.95

        # Model name for display
        model_short = "trocr-handwritten" if "kazars24" in self.model_name else "trocr-1700s"

        result = {
            "text": text,
            "confidence": round(confidence, 4),
            "model_used": model_short,
        }

        # Cache save
        if use_cache:
            self._save_cache(cache_key, result)

        return result

    def _compute_confidence(self, pixel_values, generated_ids) -> float:
        """Compute per-token confidence from logits."""
        try:
            with torch.no_grad():
                outputs = self.model(pixel_values, decoder_input_ids=generated_ids)
                logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            token_probs = []
            for i, token_id in enumerate(generated_ids[0]):
                if i < probs.shape[1]:
                    token_probs.append(probs[0, i, token_id].item())

            return float(np.mean(token_probs)) if token_probs else 0.0
        except Exception as e:
            logger.warning(f"Confidence computation failed: {e}")
            return 0.95

    # ── Tesseract fallback ──
    def recognize_with_tesseract(self, image, lang: str = None) -> dict:
        """Fallback: Tesseract OCR with Russian language."""
        if lang is None:
            lang = self.tesseract_lang

        pil_image = self._prepare_image(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            pil_image.save(temp_path)

        try:
            result = subprocess.run(
                ["tesseract", temp_path, "stdout", "-l", lang, "--psm", "3"],
                capture_output=True, text=True, timeout=30
            )
            text = result.stdout.strip()
            confidence = self._get_tesseract_confidence(temp_path, lang)
            return {
                "text": text,
                "confidence": round(confidence, 4),
                "model_used": "tesseract"
            }
        except FileNotFoundError:
            logger.error("Tesseract not installed")
            return {"text": "", "confidence": 0.0, "model_used": "error"}
        except subprocess.TimeoutExpired:
            logger.warning("Tesseract timed out")
            return {"text": "", "confidence": 0.0, "model_used": "error"}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_tesseract_confidence(self, image_path: str, lang: str) -> float:
        try:
            result = subprocess.run(
                ["tesseract", image_path, "stdout", "-l", lang, "--psm", "3", "tsv"],
                capture_output=True, text=True, timeout=30
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return 0.0
            confidences = []
            for line in lines[1:]:
                parts = line.split('\t')
                if len(parts) > 10:
                    try:
                        conf = float(parts[10])
                        if conf >= 0:
                            confidences.append(conf)
                    except (ValueError, IndexError):
                        pass
            return sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        except:
            return 0.0

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"[{self.model_name}] Resources freed")


# ── Smart Multi-Model Router ──
# Global registry of loaded engines
_engines: dict[str, OCREngine] = {}

def _get_engine(model_name: str) -> OCREngine:
    """Lazy-load and cache OCR engines."""
    if model_name not in _engines:
        _engines[model_name] = OCREngine(model_name=model_name)
    return _engines[model_name]

def recognize_text(image,
                   region_type: str = "handwritten",
                   use_voting: bool = False) -> dict:
    """Recognize Russian text from an image, auto-selecting model by region type.

    Args:
        image: PIL Image, numpy array (BGR), or file path.
        region_type: One of:
            - 'handwritten'  -> kazars24/trocr-base-handwritten-ru
            - 'printed'      -> taiga75/ru-trocr-1700s
            - 'printed_old'  -> taiga75/ru-trocr-1700s (same)
            - 'table_cell'   -> auto-detect handwritten vs printed
            - 'stamp'        -> taiga75/ru-trocr-1700s
            - 'signature'    -> kazars24/trocr-base-handwritten-ru
        use_voting: If True, also run Tesseract and pick best result.

    Returns:
        dict with keys: 'text', 'confidence', 'model_used'
    """
    # Map region types to models
    model_map = {
        "handwritten": MODEL_HANDWRITTEN,            # kazars24
        "signature": MODEL_HANDWRITTEN,              # kazars24
        "printed": MODEL_PRINTED_HISTORICAL,         # taiga75
        "printed_old": MODEL_PRINTED_HISTORICAL,     # taiga75
        "printed_modern": MODEL_PRINTED_HISTORICAL,  # taiga75 (works for both)
        "table_cell": MODEL_HANDWRITTEN,             # assume handwritten by default
        "stamp": MODEL_PRINTED_HISTORICAL,
        "marginal_note": MODEL_HANDWRITTEN,          # usually handwritten
        "text_block": MODEL_PRINTED_HISTORICAL,
        "header_row": MODEL_PRINTED_HISTORICAL,
    }

    model_name = model_map.get(region_type, MODEL_HANDWRITTEN)
    engine = _get_engine(model_name)

    # Run primary model
    result = engine.recognize(image)

    # Optional: voting with Tesseract for low-confidence results
    if use_voting and result["confidence"] < 0.6:
        tesseract_result = engine.recognize_with_tesseract(image)
        if tesseract_result["confidence"] > result["confidence"]:
            result = tesseract_result
            result["voting_note"] = "Tesseract chosen over TrOCR"
        elif len(tesseract_result.get("text", "")) > len(result.get("text", "")):
            result = tesseract_result
            result["voting_note"] = "Tesseract chosen (longer text)"

    return result

def recognize_handwritten(image, use_voting: bool = False) -> dict:
    """Shorthand: recognize handwritten text using kazars24 model."""
    return recognize_text(image, region_type="handwritten", use_voting=use_voting)

def recognize_printed(image, use_voting: bool = False) -> dict:
    """Shorthand: recognize printed text using taiga75 model."""
    return recognize_text(image, region_type="printed", use_voting=use_voting)


def recognize_text_voting(image, region_type: str = "handwritten") -> dict:
    """Recognize text with voting between TrOCR and Tesseract."""
    return recognize_text(image, region_type=region_type, use_voting=True)

def create_ocr_engine(model_name: str = MODEL_HANDWRITTEN, **kwargs) -> OCREngine:
    """Create a standalone OCR engine (not singleton)."""
    return OCREngine(model_name=model_name, **kwargs)

def preload_models():
    """Preload only printed model by default (handwritten loaded on demand)."""
    logger.info("Preloading printed model (taiga75/ru-trocr-1700s)...")
    _get_engine(MODEL_PRINTED_HISTORICAL)
    logger.info("Printed model loaded. Handwritten model will load on first use.")

def clear_all_caches(cache_dir: str = "data/cache"):
    """Clear OCR result caches."""
    for engine in _engines.values():
        engine.clear_cache(cache_dir)

def cleanup_all():
    """Free all OCR engine resources."""
    for name, engine in _engines.items():
        engine.cleanup()
    _engines.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("All OCR resources freed")