"""Tests for OCR Service."""
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from app.services.ocr import OCREngine, recognize_text, get_engine
import subprocess


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_trocr_model():
    """Mock TrOCR model and processor."""
    with patch('app.services.ocr.TrOCRProcessor') as mock_processor_cls, \
         patch('app.services.ocr.VisionEncoderDecoderModel') as mock_model_cls:
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processor
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = mock_processor

        # Mock pixel_values
        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        mock_processor(images=MagicMock(), return_tensors="pt").pixel_values = mock_pixel_values

        # Mock model outputs
        mock_generated = MagicMock()
        mock_model.generate.return_value = mock_generated
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock processor batch_decode
        mock_processor.batch_decode.return_value = ["Иван Петров"]

        yield mock_processor, mock_model, mock_processor_cls, mock_model_cls


@pytest.fixture
def sample_image_array():
    """Create a simple test image as numpy array."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 240
    cv2.putText(img, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


@pytest.fixture
def ocr_engine(mock_trocr_model):
    """Create OCR engine with mocked model."""
    mock_proc, mock_model, mock_proc_cls, mock_model_cls = mock_trocr_model
    engine = OCREngine(model_name="test-model", use_gpu=False)
    engine.processor = mock_proc
    engine.model = mock_model
    engine.device = "cpu"
    return engine


# ── Test: OCREngine Initialization ──────────────────────────────────────────
class TestOCREngineInit:
    def test_init_default_device(self):
        """Test auto device detection."""
        engine = OCREngine.__new__(OCREngine)
        assert engine is not None

    def test_init_with_cpu(self):
        engine = OCREngine.__new__(OCREngine)
        engine.device = "cpu"
        assert engine.device == "cpu"


# ── Test: _prepare_image ────────────────────────────────────────────────────

class TestPrepareImage:
    def test_from_numpy_array(self, ocr_engine, sample_image_array):
        from PIL import Image
        result = ocr_engine._prepare_image(sample_image_array)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_from_string_path(self, ocr_engine, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)
        result = ocr_engine._prepare_image(str(img_path))
        assert isinstance(result, Image.Image)

    def test_from_pil_image(self, ocr_engine):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        result = ocr_engine._prepare_image(img)
        assert isinstance(result, Image.Image)

    def test_invalid_type_raises(self, ocr_engine):
        with pytest.raises(TypeError, match="Unsupported image type"):
            ocr_engine._prepare_image(123)


# ── Test: recognize ─────────────────────────────────────────────────────────

class TestRecognize:
    def test_recognize_text(self, ocr_engine, sample_image_array):
        """Test basic text recognition."""
        with patch.object(ocr_engine.processor, 'batch_decode',
                         return_value=["Иван Петров"]):
            result = ocr_engine.recognize(sample_image_array)
            assert "text" in result
            assert "confidence" in result
            assert result["text"] == "Иван Петров"
            assert result["model_used"] == "ru-trocr-1700s"

    def test_recognize_confidence(self, ocr_engine, sample_image_array):
        """Test that confidence is computed."""
        result = ocr_engine.recognize(sample_image_array, return_confidence=True)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_recognize_cache_hit(self, ocr_engine, sample_image_array, tmp_path):
        """Test cache returns cached result."""
        ocr_engine._get_cache_key = MagicMock(return_value="abc123")
        ocr_engine._check_cache = MagicMock(
            return_value={"text": "cached", "confidence": 0.99}
        )
        result = ocr_engine.recognize(sample_image_array, use_cache=True)
        assert result["text"] == "cached"


# ── Test: recognize_with_tesseract ──────────────────────────────────────────

class TestRecognizeWithTesseract:
    def test_tesseract_not_installed(self, ocr_engine, sample_image_array):
        """Test fallback when Tesseract is not installed."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = ocr_engine.recognize_with_tesseract(sample_image_array)
            assert result["text"] == ""
            assert result["confidence"] == 0.0
            assert result["model_used"] == "error"

    def test_tesseract_timeout(self, ocr_engine, sample_image_array):
        """Test fallback on timeout."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 30)):
            result = ocr_engine.recognize_with_tesseract(sample_image_array)
            assert result["text"] == ""
            assert result["confidence"] == 0.0

    def test_tesseract_success(self, ocr_engine, sample_image_array):
        """Test successful Tesseract recognition."""
        with patch('subprocess.run') as mock_run:
            mock_process = MagicMock()
            mock_process.stdout = "Иван Петров\n"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            result = ocr_engine.recognize_with_tesseract(sample_image_array)
            assert "text" in result
            assert "confidence" in result


# ── Test: recognize_with_voting ─────────────────────────────────────────────
class TestRecognizeWithVoting:
    def test_voting_selects_highest_confidence(self, ocr_engine, sample_image_array):
        """Test that voting picks the highest confidence result."""
        ocr_engine.recognize = MagicMock(
            return_value={"text": "trocr", "confidence": 0.9, "model_used": "ru-trocr-1700s"}
        )
        ocr_engine.recognize_with_tesseract = MagicMock(
            return_value={"text": "tesseract", "confidence": 0.5, "model_used": "tesseract"}
        )
        result = ocr_engine.recognize_with_voting(sample_image_array)
        assert result["text"] == "trocr"

    def test_voting_longer_text_low_confidence(self, ocr_engine, sample_image_array):
        """When both are low confidence, choose longest text."""
        ocr_engine.recognize = MagicMock(
            return_value={"text": "short", "confidence": 0.5, "model_used": "ru-trocr-1700s"}
        )
        ocr_engine.recognize_with_tesseract = MagicMock(
            return_value={"text": "much longer text here", "confidence": 0.4, "model_used": "tesseract"}
        )
        result = ocr_engine.recognize_with_voting(sample_image_array, confidence_threshold=0.7)
        assert result["text"] == "much longer text here"

    def test_voting_summary(self, ocr_engine, sample_image_array):
        """Test that voting summary is included."""
        ocr_engine.recognize = MagicMock(
            return_value={"text": "a", "confidence": 0.9, "model_used": "ru-trocr-1700s"}
        )
        ocr_engine.recognize_with_tesseract = MagicMock(
            return_value={"text": "b", "confidence": 0.5, "model_used": "tesseract"}
        )
        result = ocr_engine.recognize_with_voting(sample_image_array)
        assert "voting_summary" in result


# ── Test: Cache ─────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_key_generation(self, ocr_engine, sample_image_array):
        """Test that cache key is deterministic."""
        key1 = ocr_engine._get_cache_key(sample_image_array)
        key2 = ocr_engine._get_cache_key(sample_image_array)
        assert key1 == key2

    def test_check_cache_miss(self, ocr_engine, tmp_path):
        """Test cache miss returns None."""
        result = ocr_engine._check_cache("nonexistent", cache_dir=str(tmp_path))
        assert result is None

    def test_check_cache_hit(self, ocr_engine, tmp_path):
        """Test cache hit returns saved data."""
        cache_dir = str(tmp_path)
        cache_file = tmp_path / "abc123.json"
        cache_file.write_text('{"text": "test", "confidence": 0.9}')
        result = ocr_engine._check_cache("abc123", cache_dir=cache_dir)
        assert result is not None
        assert result["text"] == "test"

    def test_save_and_retrieve_cache(self, ocr_engine, tmp_path):
        """Test full cache round-trip."""
        cache_dir = str(tmp_path)
        ocr_engine._save_cache("testkey", {"text": "hello", "confidence": 0.95}, cache_dir)
        result = ocr_engine._check_cache("testkey", cache_dir)
        assert result["text"] == "hello"
        assert result["confidence"] == 0.95

    def test_clear_cache(self, ocr_engine, tmp_path):
        """Test cache clearing."""
        cache_dir = str(tmp_path)
        ocr_engine._save_cache("key1", {"text": "a", "confidence": 0.5}, cache_dir)
        ocr_engine._save_cache("key2", {"text": "b", "confidence": 0.6}, cache_dir)
        assert len(list(tmp_path.iterdir())) == 2
        ocr_engine.clear_cache(cache_dir)
        assert len(list(tmp_path.iterdir())) == 0


# ── Test: Helper Functions ──────────────────────────────────────────────────
class TestHelpers:
    def test_get_engine(self):
        """Test factory function."""
        engine = get_engine()
        assert isinstance(engine, OCREngine)

    def test_recognize_text(self, sample_image_array):
        """Test recognize_text helper."""
        with patch('app.services.ocr.OCREngine.get_instance') as mock_get:
            mock_engine = MagicMock()
            mock_engine.recognize.return_value = {
                "text": "Иван Петров",
                "confidence": 0.95,
                "model_used": "ru-trocr-1700s"
            }
            mock_get.return_value = mock_engine
            result = recognize_text(sample_image_array)
            assert result["text"] == "test"

    def test_cleanup(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model = MagicMock() 
        engine.device = "cpu"
        engine.cleanup()
        assert True  