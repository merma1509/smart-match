"""Tests for light_preprocess.py — Stage 1 preprocessing before layout detection."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from app.services.light_preprocess import (
    light_preprocess,
    _deskew,
    _limit_image_size,
    _color_normalize,
    _mild_denoise,
    _illumination_normalize,
    _remove_borders,
    MAX_IMAGE_DIMENSION,
)


# ── Find images dynamically ──

def _find_test_images() -> list[Path]:
    """Search for .jpg files in data/ recursively."""
    data_dir = Path("data/01-0203-0745-000600")
    if not data_dir.exists():
        return []
    images = sorted(data_dir.rglob("*.jpg"))
    # Exclude test_output
    images = [p for p in images if "test_output" not in str(p)]
    return images


ALL_TEST_IMAGES = _find_test_images()
TEST_OUTPUT_DIR = Path("data/test_output")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n Found {len(ALL_TEST_IMAGES)} test images")
if ALL_TEST_IMAGES:
    print(f"  First: {ALL_TEST_IMAGES[0]}")
    print(f"  Last:  {ALL_TEST_IMAGES[-1]}\n")


# ── Fixtures ──

@pytest.fixture(scope="session")
def output_dir() -> Path:
    return TEST_OUTPUT_DIR


@pytest.fixture(params=ALL_TEST_IMAGES[:3] if ALL_TEST_IMAGES else [""])
def sample_image_path(request) -> Path:
    path = request.param
    if not path:
        pytest.skip("No test images found in data/")
    return path


@pytest.fixture
def sample_image(sample_image_path) -> np.ndarray:
    img = cv2.imread(str(sample_image_path))
    if img is None:
        pytest.skip(f"Cannot load {sample_image_path}")
    return img


@pytest.fixture
def sample_image_name(sample_image_path) -> str:
    return sample_image_path.stem


@pytest.fixture
def large_image() -> np.ndarray:
    """Use the largest image (by file size)."""
    if not ALL_TEST_IMAGES:
        pytest.skip("No images found")
    largest = max(ALL_TEST_IMAGES, key=lambda p: p.stat().st_size)
    img = cv2.imread(str(largest))
    if img is None:
        pytest.skip(f"Cannot load {largest}")
    return img


@pytest.fixture
def small_image() -> np.ndarray:
    """Use the smallest image (by file size)."""
    if not ALL_TEST_IMAGES:
        pytest.skip("No images found")
    smallest = min(ALL_TEST_IMAGES, key=lambda p: p.stat().st_size)
    img = cv2.imread(str(smallest))
    if img is None:
        pytest.skip(f"Cannot load {smallest}")
    return img


# ── Helpers ──

def save_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    name: str,
    output_dir: Path,
    step: str = "",
) -> Path:
    h_orig, w_orig = original.shape[:2]
    h_proc, w_proc = processed.shape[:2]
    target_h = min(h_orig, h_proc, 800)
    orig_resized = cv2.resize(original, (int(w_orig * target_h / h_orig), target_h))
    proc_resized = cv2.resize(processed, (int(w_proc * target_h / h_proc), target_h))
    gap = 10
    total_w = orig_resized.shape[1] + proc_resized.shape[1] + gap
    canvas = np.ones((target_h, total_w, 3), dtype=np.uint8) * 200
    canvas[:, :orig_resized.shape[1]] = orig_resized
    canvas[:, orig_resized.shape[1] + gap:] = proc_resized
    label_prefix = f"{name}_{step}" if step else name
    out_path = output_dir / f"{label_prefix}_comparison.jpg"
    cv2.putText(canvas, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, f"Processed ({step or 'full'})",
                (orig_resized.shape[1] + gap + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), canvas)
    print(f"\n  Saved: {out_path}")
    return out_path


# ── Unit Tests ──

class TestLimitImageSize:
    def test_resizes_large_image(self, large_image):
        result = _limit_image_size(large_image, max_dim=MAX_IMAGE_DIMENSION)
        h, w = result.shape[:2]
        assert max(h, w) <= MAX_IMAGE_DIMENSION

    def test_preserves_aspect_ratio(self, large_image):
        orig_ratio = large_image.shape[0] / large_image.shape[1]
        result = _limit_image_size(large_image, max_dim=MAX_IMAGE_DIMENSION)
        new_ratio = result.shape[0] / result.shape[1]
        assert abs(orig_ratio - new_ratio) < 0.01

    def test_output_is_color(self, large_image):
        result = _limit_image_size(large_image, max_dim=MAX_IMAGE_DIMENSION)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_small_image_unchanged(self, small_image):
        h, w = small_image.shape[:2]
        if max(h, w) <= MAX_IMAGE_DIMENSION:
            result = _limit_image_size(small_image)
            assert result.shape == small_image.shape


class TestDeskew:
    def test_preserves_color(self, sample_image):
        result = _deskew(sample_image)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_output_not_empty(self, sample_image):
        result = _deskew(sample_image)
        assert np.sum(result) > 0

    def test_saves_visualization(self, sample_image, sample_image_name, output_dir):
        result = _deskew(sample_image)
        save_comparison(sample_image, result, sample_image_name, output_dir, "deskew")


class TestColorNormalize:
    def test_preserves_content(self, sample_image):
        result = _color_normalize(sample_image)
        assert result.shape == sample_image.shape
        assert np.sum(result) > 0
        assert result.dtype == np.uint8

    def test_output_is_uint8(self, sample_image):
        result = _color_normalize(sample_image)
        assert result.dtype == np.uint8

    def test_saves_visualization(self, sample_image, sample_image_name, output_dir):
        result = _color_normalize(sample_image)
        save_comparison(sample_image, result, sample_image_name, output_dir, "color_norm")


class TestMildDenoise:
    def test_output_same_shape(self, sample_image):
        result = _mild_denoise(sample_image)
        assert result.shape == sample_image.shape

    def test_output_is_color(self, sample_image):
        result = _mild_denoise(sample_image)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_saves_visualization(self, sample_image, sample_image_name, output_dir):
        result = _mild_denoise(sample_image)
        save_comparison(sample_image, result, sample_image_name, output_dir, "denoise")


class TestIlluminationNormalize:
    def test_output_is_color(self, sample_image):
        result = _illumination_normalize(sample_image)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_saves_visualization(self, sample_image, sample_image_name, output_dir):
        result = _illumination_normalize(sample_image)
        save_comparison(sample_image, result, sample_image_name, output_dir, "illumination")


class TestRemoveBorders:
    def test_clears_edges(self, sample_image):
        result = _remove_borders(sample_image, border_width=5)
        assert np.all(result[0, 0] >= 200)


# ── Integration Tests ──

class TestLightPreprocess:
    def test_full_pipeline_no_error(self, sample_image):
        result = light_preprocess(sample_image)
        assert result is not None

    def test_output_is_color(self, sample_image):
        result = light_preprocess(sample_image)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_output_dtype_uint8(self, sample_image):
        result = light_preprocess(sample_image)
        assert result.dtype == np.uint8

    def test_does_not_binarize(self, sample_image):
        result = light_preprocess(sample_image)
        unique_values = len(np.unique(result))
        assert unique_values > 10

    def test_aspect_ratio_preserved(self, sample_image):
        result = light_preprocess(sample_image)
        orig_ratio = sample_image.shape[0] / sample_image.shape[1]
        new_ratio = result.shape[0] / result.shape[1]
        assert round(abs(orig_ratio - new_ratio),2) < 0.20

    def test_saves_visualization(self, sample_image, sample_image_name, output_dir):
        result = light_preprocess(sample_image)
        save_comparison(sample_image, result, sample_image_name, output_dir, "full_pipeline")


class TestStepByStepVisualization:
    STEPS = [
        ("01_resize", lambda img: _limit_image_size(img)),
        ("02_denoise", lambda img: _mild_denoise(img)),
        ("03_color_normalize", lambda img: _color_normalize(img)),
        ("04_illumination_normalize", lambda img: _illumination_normalize(img)),
        ("05_deskew", lambda img: _deskew(img)),
        ("06_border_removal", lambda img: _remove_borders(img)),
    ]

    def test_all_steps_visual(self, sample_image, sample_image_name, output_dir):
        current = sample_image.copy()
        for step_name, step_fn in self.STEPS:
            result = step_fn(current)
            save_comparison(current, result, sample_image_name, output_dir, step_name)
            current = result


# ── Batch Test ──

class TestBatchProcessing:
    def test_all_images_process(self, output_dir):
        if not ALL_TEST_IMAGES:
            pytest.skip("No images found")
        failures = []
        for img_path in ALL_TEST_IMAGES:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    failures.append(f"{img_path.name}: cannot read")
                    continue
                result = light_preprocess(img)
                if result is None:
                    failures.append(f"{img_path.name}: returned None")
                elif len(result.shape) != 3 or result.shape[2] != 3:
                    failures.append(f"{img_path.name}: non-color {result.shape}")
            except Exception as e:
                failures.append(f"{img_path.name}: {e}")
        assert not failures, (
            f"Failed on {len(failures)}/{len(ALL_TEST_IMAGES)}:\n" +
            "\n".join(failures[:10])
        )


# ── Config Tests ──

class TestConfig:
    def test_max_image_dimension_reasonable(self):
        assert 1000 <= MAX_IMAGE_DIMENSION <= 4000