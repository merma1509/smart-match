# Business logic: OCR — Russian historical text recognition
# Primary: taiga75/ru-trocr-1700s for Russian handwriting and printed text
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


RUSSIAN_MODEL = "taiga75/ru-trocr-1700s"


class OCREngine:
    _engine_instance = None
    """OCR engine for Russian historical texts.
    Primary: ru-trocr-1700s for Russian text.
    Fallback: Tesseract with Russian language.
    Singleton pattern for model reuse.
    """

    @classmethod
    def get_instance(cls, **kwargs):
        """Get or create the singleton OCR engine instance [1]."""
        if cls._engine_instance is None:
            cls._engine_instance = cls(**kwargs)
        return cls._engine_instance

    """OCR engine for Russian historical texts.
    
    Primary: ru-trocr-1700s for Cyrillic handwriting and printed text.
    Fallback: Tesseract with Russian language.
    """
    
    def __init__(self, 
                 model_name: str = RUSSIAN_MODEL,
                 tesseract_lang: str = "rus",
                 device: str = None, 
                 use_gpu: bool = True):
        
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
        logger.info(f"Using device: {self.device}")
        
        # Load Russian model
        logger.info(f"Loading model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        logger.info(f"OCR Engine initialized (model={model_name}, tesseract={tesseract_lang})")
    
    def _prepare_image(self, image) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def recognize(self, image, return_confidence: bool = True, use_cache: bool = True) -> dict:
        """Recognize Russian text from an image using ru-trocr-1700s."""
        pil_image = self._prepare_image(image)

        if use_cache:
            cache_key = self._get_cache_key(pil_image)
            cached = self._check_cache(cache_key)
            if cached:
                return cached
        
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        confidence = 0.95
        if return_confidence:
            confidence, _ = self._compute_confidence(pixel_values, generated_ids)
        
        result = {
            "text": text,
            "confidence": round(confidence, 4),
            "model_used": "ru-trocr-1700s",
        }
        
        if use_cache:
            self._save_cache(cache_key, result)
        
        return result
    
    def _compute_confidence(self, pixel_values, generated_ids) -> tuple:
        try:
            with torch.no_grad():
                outputs = self.model(pixel_values, decoder_input_ids=generated_ids)
                logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            token_probs = []
            
            for i, token_id in enumerate(generated_ids[0]):
                if i < probs.shape[1]:
                    token_probs.append(probs[0, i, token_id].item())
            
            return float(np.mean(token_probs)) if token_probs else 0.0, []
        except Exception as e:
            logger.warning(f"Confidence computation failed: {e}")
            return 0.95, []
    
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
            
            return {"text": text, "confidence": round(confidence, 4), "model_used": "tesseract"}
            
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
    
    def recognize_with_voting(self, image, confidence_threshold: float = 0.7) -> dict:
        """Vote between ru-trocr-1700s and Tesseract for best Russian result."""
        trocr_result = self.recognize(image)
        tesseract_result = self.recognize_with_tesseract(image)
        
        # Pick highest confidence
        if trocr_result["confidence"] >= tesseract_result["confidence"]:
            best = trocr_result
        else:
            best = tesseract_result
        
        # If both low, choose longer text
        if best["confidence"] < confidence_threshold:
            longer = max([trocr_result, tesseract_result], key=lambda x: len(x.get("text", "")))
            best = longer
            best["voting_note"] = "Low confidence - chose longest text"
        else:
            best["voting_note"] = f"Best confidence"
        
        best["voting_summary"] = {
            "ru_trocr_confidence": trocr_result["confidence"],
            "tesseract_confidence": tesseract_result["confidence"],
        }
        
        return best
    
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
    
    def cleanup(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("OCR engine resources freed")


def create_ocr_engine(**kwargs) -> OCREngine:
    return OCREngine(**kwargs)

def recognize_text(image, engine=None):
    """Helper function: recognize text from image using default OCR engine [1]."""
    if engine is None:
        engine = OCREngine.get_instance()
    return engine.recognize(image)


def recognize_text_voting(image, engine=None):
    """Helper function: recognize text with voting between ru-trocr and tesseract."""
    if engine is None:
        engine = OCREngine()
    return engine.recognize_with_voting(image)
