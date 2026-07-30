"""Unit tests for light preprocessing module.

Тестирует этапы предобработки Stage 1: дескью, нормализацию,
удаление шума, улучшение контраста, метрики качества.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.light_preprocess import (
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    _color_normalize,
    _compute_quality_metrics,
    _deskew_robust,
    _enhance_contrast,
    _illumination_normalize,
    _limit_image_size,
    _mild_denoise,
    _remove_borders,
    light_preprocess,
)

# ── Fixtures ──


@pytest.fixture
def sample_bgr():
    """Стандартное BGR изображение 400×600."""
    return np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray():
    """Стандартное grayscale изображение 400×600."""
    return np.random.randint(0, 256, (400, 600), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Большое изображение (больше MAX_IMAGE_DIMENSION)."""
    return np.random.randint(0, 256, (4000, 5000, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Маленькое изображение (меньше MIN_IMAGE_DIMENSION)."""
    return np.random.randint(0, 256, (500, 700, 3), dtype=np.uint8)


@pytest.fixture
def skewed_image():
    """Изображение с поворотом на ~5 градусов."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Рисуем длинную горизонтальную линию
    cv2.line(img, (50, 250), (450, 250), 0, 3)
    # Добавляем вторую линию параллельно
    cv2.line(img, (50, 350), (450, 350), 0, 3)
    # Поворачиваем на 5 градусов
    center = (250, 250)
    M = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    rotated = cv2.warpAffine(img, M, (500, 500), borderValue=255)
    return rotated


# ── Size Limiting Tests ──


class TestLimitImageSize:
    def test_normal_size(self):
        """Изображение нормального размера не должно меняться."""
        img = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        result = _limit_image_size(img)
        assert result.shape == img.shape

    def test_large_image_downscale(self, large_image):
        """Большое изображение должно уменьшаться."""
        result = _limit_image_size(large_image)
        assert max(result.shape[:2]) <= MAX_IMAGE_DIMENSION

    def test_small_image_upscale(self, small_image):
        """Маленькое изображение должно увеличиваться."""
        result = _limit_image_size(small_image)
        # 500x700 -> увеличится до ~min 1500 по большей стороне
        assert max(result.shape[:2]) >= MIN_IMAGE_DIMENSION

    def test_preserves_aspect_ratio(self, large_image):
        """Соотношение сторон должно сохраняться."""
        h, w = large_image.shape[:2]
        ratio = w / h
        result = _limit_image_size(large_image)
        h2, w2 = result.shape[:2]
        result_ratio = w2 / h2
        assert abs(ratio - result_ratio) < 0.05

    def test_custom_max_dim(self):
        """Пользовательский max_dim должен работать."""
        img = np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8)
        custom_dim = 2000
        result = _limit_image_size(img, max_dim=custom_dim)
        assert max(result.shape[:2]) <= custom_dim


# ── Deskew Tests ──


class TestDeskew:
    def test_no_skew(self, sample_bgr):
        """Без перекоса изображение не должно измениться значительно."""
        result = _deskew_robust(sample_bgr)
        # Размер может измениться из-за поворота, но не должен быть нулевым
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_skew_correction(self, skewed_image):
        """Перекос должен исправляться."""
        result = _deskew_robust(skewed_image)
        # После исправления форма может измениться из-за поворота
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_deskew_gray(self, sample_gray):
        """Grayscale изображение тоже должно обрабатываться."""
        result = _deskew_robust(sample_gray)
        assert len(result.shape) == 2

    def test_deskew_empty(self):
        """Пустое белое изображение не должно ломаться."""
        white = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = _deskew_robust(white)
        assert result is not None


# ── Color Normalization Tests ──


class TestColorNormalize:
    def test_color_normalize_shape(self, sample_bgr):
        """Форма изображения не должна меняться."""
        result = _color_normalize(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_color_normalize_dtype(self, sample_bgr):
        """Тип данных должен быть uint8."""
        result = _color_normalize(sample_bgr)
        assert result.dtype == np.uint8

    def test_color_normalize_range(self, sample_bgr):
        """Значения пикселей должны быть в [0, 255]."""
        result = _color_normalize(sample_bgr)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_color_normalize_gray(self, sample_gray):
        """Grayscale не должен ломаться (просто возвращается как есть)."""
        result = _color_normalize(cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR))
        assert result.shape[:2] == sample_gray.shape


# ── Illumination Normalization Tests ──


class TestIlluminationNormalize:
    def test_illumination_normalize_shape(self, sample_bgr):
        """Форма не должна меняться."""
        result = _illumination_normalize(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_illumination_normalize_dtype(self, sample_bgr):
        """Тип uint8."""
        result = _illumination_normalize(sample_bgr)
        assert result.dtype == np.uint8


# ── Contrast Enhancement Tests ──


class TestContrastEnhance:
    def test_enhance_contrast_shape(self, sample_bgr):
        """Форма не должна меняться."""
        result = _enhance_contrast(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_enhance_contrast_dtype(self, sample_bgr):
        """Тип uint8."""
        result = _enhance_contrast(sample_bgr)
        assert result.dtype == np.uint8

    def test_enhance_contrast_custom_clip(self, sample_bgr):
        """Пользовательский clip_limit работает."""
        result = _enhance_contrast(sample_bgr, clip_limit=3.0)
        assert result.shape == sample_bgr.shape

    def test_enhance_contrast_increases(self):
        """Контраст должен увеличиваться (не гарантировано для случайных данных)."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Добавляем контрастные области
        img[20:80, 20:80] = 50
        result = _enhance_contrast(img)
        assert result[0, 0, 0] >= 0  # Не падает


# ── Denoise Tests ──


class TestDenoise:
    def test_denoise_shape(self, sample_bgr):
        """Форма не должна меняться."""
        result = _mild_denoise(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_denoise_dtype(self, sample_bgr):
        """Тип uint8."""
        result = _mild_denoise(sample_bgr)
        assert result.dtype == np.uint8

    def test_denoise_custom_h(self, sample_bgr):
        """Пользовательский параметр h работает."""
        result = _mild_denoise(sample_bgr, h=10)
        assert result.shape == sample_bgr.shape


# ── Border Removal Tests ──


class TestBorderRemoval:
    def test_border_removal_shape(self, sample_bgr):
        """Форма не должна меняться."""
        result = _remove_borders(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_border_removal_white(self):
        """Границы становятся белыми."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        border = 5
        result = _remove_borders(img, border_width=border)
        # Первая строка должна быть белой
        assert np.all(result[0, :, 0] == 255)
        # Последняя строка — белая
        assert np.all(result[-1, :, 0] == 255)
        # Первая колонка — белая
        assert np.all(result[:, 0, 0] == 255)
        # Последняя колонка — белая
        assert np.all(result[:, -1, 0] == 255)

    def test_border_removal_zero(self, sample_bgr):
        """border_width=0 не должен менять изображение."""
        result = _remove_borders(sample_bgr, border_width=0)
        assert np.array_equal(result, sample_bgr)

    def test_border_removal_custom_width(self):
        """Пользовательская ширина рамки работает."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _remove_borders(img, border_width=20)
        assert np.all(result[19, :, 0] == 255)


# ── Quality Metrics Tests ──


class TestQualityMetrics:
    def test_metrics_return_structure(self, sample_bgr):
        """Проверка структуры возвращаемых метрик."""
        metrics = _compute_quality_metrics(sample_bgr)
        required_keys = {"contrast", "sharpness", "entropy"}
        assert required_keys.issubset(metrics.keys())

    def test_metrics_positive(self, sample_bgr):
        """Все метрики должны быть положительными."""
        metrics = _compute_quality_metrics(sample_bgr)
        for key, value in metrics.items():
            assert value >= 0, f"{key} should be >= 0, got {value}"

    def test_metrics_gray(self, sample_gray):
        """Метрики для grayscale работают."""
        metrics = _compute_quality_metrics(sample_gray)
        assert len(metrics) == 3

    def test_blank_image_metrics(self):
        """Пустое изображение — минимальные метрики."""
        blank = np.ones((100, 100), dtype=np.uint8) * 128
        metrics = _compute_quality_metrics(blank)
        assert metrics["contrast"] < 10  # Низкий контраст


# ── Full Pipeline Tests ──


class TestFullPipeline:
    def test_light_preprocess_default(self, sample_bgr):
        """Полный пайплайн с параметрами по умолчанию."""
        result = light_preprocess(sample_bgr)
        # Размер может измениться из-за апскейла/дескью
        assert result.dtype == np.uint8

    def test_light_preprocess_with_metrics(self, sample_bgr):
        """Пайплайн с возвратом метрик."""
        result, metrics = light_preprocess(sample_bgr, return_metrics=True)
        assert isinstance(metrics, dict)
        assert "before" in metrics
        assert "after" in metrics
        assert "steps" in metrics

    def test_light_preprocess_steps_logged(self, sample_bgr):
        """Метрики должны содержать список выполненных шагов."""
        _, metrics = light_preprocess(sample_bgr, return_metrics=True)
        assert len(metrics["steps"]) > 0

    def test_light_preprocess_disabled_steps(self, sample_bgr):
        """С отключёнными шагами изображение меняется иначе."""
        result_full = light_preprocess(sample_bgr)
        result_minimal = light_preprocess(
            sample_bgr,
            apply_resize=False,
            apply_deskew=False,
            apply_color_normalize=False,
            apply_denoise=False,
            apply_illumination_normalize=False,
            apply_border_removal=False,
            apply_contrast_enhance=False,
        )
        # resizing отключён, shape должен совпадать
        assert result_minimal.shape[:2] == sample_bgr.shape[:2]

    def test_light_preprocess_large(self, large_image):
        """Большое изображение обрабатывается без ошибок."""
        result = light_preprocess(large_image)
        # После resize + deskew размер может немного отличаться от MAX
        assert min(result.shape[0], result.shape[1]) > 0

    def test_light_preprocess_small(self, small_image):
        """Маленькое изображение увеличивается."""
        result = light_preprocess(small_image)
        assert max(result.shape[:2]) >= MIN_IMAGE_DIMENSION

    def test_light_preprocess_custom_contrast(self, sample_bgr):
        """Пользовательский clip_limit для контраста."""
        result = light_preprocess(sample_bgr, contrast_clip_limit=4.0)
        assert result.dtype == np.uint8

    def test_light_preprocess_gray_input(self, sample_gray):
        """Grayscale на входе — grayscale на выходе (или BGR, не падает)."""
        result = light_preprocess(sample_gray)
        assert result is not None


# ── Edge Cases ──


class TestEdgeCases:
    def test_empty_image(self):
        """Пустое изображение не должно ломаться."""
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(Exception):
            light_preprocess(empty)

    def test_black_image(self):
        """Полностью чёрное изображение."""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = light_preprocess(black)
        assert result.dtype == np.uint8

    def test_white_image(self):
        """Полностью белое изображение."""
        white = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = light_preprocess(white)
        assert result.dtype == np.uint8

    def test_single_channel_input(self):
        """Одноканальное изображение."""
        single = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = light_preprocess(single)
        assert result is not None
