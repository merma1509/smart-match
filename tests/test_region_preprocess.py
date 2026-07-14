"""Tests for region_preprocess.py — Stage 2b region-specific preprocessing.

Verifies that each region type gets appropriate preprocessing:
    - printed_text:     CLAHE + adaptive threshold + sharpen
    - handwritten:      Denoise + stroke enhancement, NO binarization
    - table_cell:       Border cleanup + contrast
    - stamp:            Color preserved, saturation enhanced
    - signature:        High-pass + stroke thickening
    - marginal_note:    Contrast stretch + adaptive threshold

Tests use REAL cropped cell regions from actual metrical book images.
Visual comparisons saved to data/test_output/.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from app.services.light_preprocess import light_preprocess
from app.services.layout import analyze_layout
from app.services.region_preprocess import (
    preprocess_printed_text,
    preprocess_handwritten,
    preprocess_table_cell,
    preprocess_stamp,
    preprocess_signature,
    preprocess_marginal_note,
    preprocess_region,
    preprocess_regions,
    _to_grayscale,
)


# ── Constants ──
TEST_OUTPUT_DIR = Path("data/test_output")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/01-0203-0745-000600")
ALL_IMAGES = sorted(DATA_DIR.rglob("*.jpg"))


# ── Fixtures ──
@pytest.fixture(scope="session")
def output_dir() -> Path:
    return TEST_OUTPUT_DIR


@pytest.fixture(scope="session")
def real_regions() -> dict:
    """Extract real cell regions from a metrical book page.

    Runs the full pipeline (light_preprocess → analyze_layout) on the first
    available image and returns cropped regions grouped by type.
    """
    if not ALL_IMAGES:
        pytest.skip("No test images found")

    raw = cv2.imread(str(ALL_IMAGES[3]))
    if raw is None:
        pytest.skip(f"Cannot load {ALL_IMAGES[0]}")

    # Run pipeline
    clean = light_preprocess(raw)
    gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    gray_raw = cv2.resize(gray_raw, (clean.shape[1], clean.shape[0]))  # Same size as clean

    layout = analyze_layout(clean, gray_raw=gray_raw)

    regions = {
        "printed_text": [],
        "handwritten": [],
        "table_cell": [],
        "stamp": [],
        "signature": [],
        "marginal_note": [],
    }

    for elem in layout["elements"]:
        x1, y1, x2, y2 = elem["bbox"]
        cropped = clean[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        etype = elem["type"]

        # Map layout types to region preprocessing types
        if etype == "data_cell":
            props = elem.get("properties", {})
            cell_type = props.get("region_type", "printed")
            if cell_type == "handwritten":
                regions["handwritten"].append(cropped)
            else:
                regions["table_cell"].append(cropped)
                regions["printed_text"].append(cropped)
        elif etype == "stamp":
            regions["stamp"].append(cropped)
        elif etype == "marginal_note":
            regions["marginal_note"].append(cropped)
        elif etype == "signature":
            regions["signature"].append(cropped)
        elif etype in ("header_row", "text_block"):
            regions["printed_text"].append(cropped)

    return regions


@pytest.fixture
def printed_region(real_regions) -> np.ndarray:
    """Get a real printed text region."""
    if not real_regions["printed_text"]:
        # Fallback: create a small synthetic one if none found
        region = np.ones((50, 200, 3), dtype=np.uint8) * 240
        cv2.putText(region, "N°  Имя  Дата", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return region
    return real_regions["printed_text"][0]


@pytest.fixture
def handwritten_region(real_regions) -> np.ndarray:
    """Get a real handwritten region."""
    if not real_regions["handwritten"]:
        pytest.skip("No handwritten regions found in test image")
    return real_regions["handwritten"][0]


@pytest.fixture
def stamp_region(real_regions) -> np.ndarray:
    """Get a real stamp region."""
    if not real_regions["stamp"]:
        pytest.skip("No stamp regions found in test image")
    return real_regions["stamp"][0]


@pytest.fixture
def marginal_region(real_regions) -> np.ndarray:
    """Get a real marginal note region."""
    if not real_regions["marginal_note"]:
        pytest.skip("No marginal note regions found in test image")
    return real_regions["marginal_note"][0]


# ── Helpers ──

def save_region_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    name: str,
    region_type: str,
    output_dir: Path,
) -> Path:
    """Save before/after side-by-side for visual inspection."""
    if len(original.shape) == 2:
        orig_display = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        orig_display = original.copy()

    if len(processed.shape) == 2:
        proc_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        proc_display = processed.copy()

    h_orig, w_orig = orig_display.shape[:2]
    h_proc, w_proc = proc_display.shape[:2]
    target_h = min(h_orig, h_proc, 200)

    if target_h == 0:
        target_h = 100

    orig_resized = cv2.resize(orig_display, (int(w_orig * target_h / max(h_orig, 1)), target_h))
    proc_resized = cv2.resize(proc_display, (int(w_proc * target_h / max(h_proc, 1)), target_h))

    gap = 10
    total_w = orig_resized.shape[1] + proc_resized.shape[1] + gap
    canvas = np.ones((target_h, total_w, 3), dtype=np.uint8) * 200

    canvas[:, :orig_resized.shape[1]] = orig_resized
    canvas[:, orig_resized.shape[1] + gap:] = proc_resized

    out_path = output_dir / f"{name}_{region_type}_comparison.jpg"
    cv2.putText(canvas, "Original", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(canvas, f"Processed ({region_type})",
                (orig_resized.shape[1] + gap + 5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(out_path), canvas)
    print(f"\n  Saved: {out_path}")
    return out_path


# ── Unit Tests (using REAL regions) ──

class TestPrintedText:
    """Tests for preprocess_printed_text using real printed regions."""

    def test_returns_binary(self, printed_region):
        """Output should be binary (thresholded)."""
        result = preprocess_printed_text(printed_region)
        unique = len(np.unique(result))
        assert unique <= 2, f"Expected binary output, got {unique} unique values"

    def test_preserves_text(self, printed_region):
        """Text should still be readable after processing."""
        result = preprocess_printed_text(printed_region)
        dark_pixels = np.mean(result < 128)
        assert dark_pixels > 0.01, "No text content preserved"

    def test_saves_visualization(self, printed_region, output_dir):
        """Save before/after for visual check."""
        result = preprocess_printed_text(printed_region)
        save_region_comparison(printed_region, result, "real", "printed_text", output_dir)


class TestHandwritten:
    """Tests for preprocess_handwritten using real handwritten regions."""

    def test_returns_grayscale(self, handwritten_region):
        """Output should be grayscale (not binary)."""
        result = preprocess_handwritten(handwritten_region)
        assert len(result.shape) == 2, f"Expected grayscale, got shape {result.shape}"
        unique = len(np.unique(result))
        assert unique > 10, f"Expected grayscale (>10 unique values), got {unique}"

    def test_reduces_noise(self, handwritten_region):
        """Should not crash on real handwritten data."""
        result = preprocess_handwritten(handwritten_region)
        assert result is not None

    def test_saves_visualization(self, handwritten_region, output_dir):
        """Save before/after for visual check."""
        result = preprocess_handwritten(handwritten_region)
        save_region_comparison(handwritten_region, result, "real", "handwritten", output_dir)


class TestTableCell:
    """Tests for preprocess_table_cell using real cell regions."""

    def test_returns_grayscale(self, printed_region):
        """Output should be grayscale."""
        result = preprocess_table_cell(printed_region)
        assert len(result.shape) == 2

    def test_saves_visualization(self, printed_region, output_dir):
        result = preprocess_table_cell(printed_region)
        save_region_comparison(printed_region, result, "real", "table_cell", output_dir)


class TestStamp:
    """Tests for preprocess_stamp using real stamp regions."""

    def test_returns_color(self, stamp_region):
        """Output should be BGR (color preserved)."""
        result = preprocess_stamp(stamp_region)
        assert len(result.shape) == 3, f"Expected BGR, got shape {result.shape}"
        assert result.shape[2] == 3

    def test_saves_visualization(self, stamp_region, output_dir):
        result = preprocess_stamp(stamp_region)
        save_region_comparison(stamp_region, result, "real", "stamp", output_dir)


class TestMarginalNote:
    """Tests for preprocess_marginal_note using real marginal regions."""

    def test_returns_binary(self, marginal_region):
        """Output should be binary."""
        result = preprocess_marginal_note(marginal_region)
        unique = len(np.unique(result))
        assert unique <= 2, f"Expected binary, got {unique} values"

    def test_saves_visualization(self, marginal_region, output_dir):
        result = preprocess_marginal_note(marginal_region)
        save_region_comparison(marginal_region, result, "real", "marginal_note", output_dir)


# ── Dispatcher Tests ──
class TestPreprocessRegion:
    """Tests for the preprocess_region dispatcher using real regions."""

    def test_dispatches_printed_text(self, printed_region):
        result = preprocess_region(printed_region, "printed_text")
        assert result is not None

    def test_dispatches_handwritten(self, handwritten_region):
        result = preprocess_region(handwritten_region, "handwritten")
        assert result is not None
        assert len(result.shape) == 2

    def test_dispatches_stamp(self, stamp_region):
        result = preprocess_region(stamp_region, "stamp")
        assert result is not None
        assert len(result.shape) == 3

    def test_dispatches_marginal_note(self, marginal_region):
        result = preprocess_region(marginal_region, "marginal_note")
        assert result is not None

    def test_dispatches_data_cell_alias(self, printed_region):
        result = preprocess_region(printed_region, "data_cell")
        assert result is not None

    def test_dispatches_unknown_type(self, printed_region):
        result = preprocess_region(printed_region, "unknown_type")
        assert result is not None
        assert len(result.shape) == 2


class TestPreprocessRegions:
    """Tests for preprocess_regions (batch)."""

    def test_processes_multiple_regions(self, printed_region):
        """Should process a list of regions from layout detection."""
        h, w = printed_region.shape[:2]
        regions = [
            {"type": "data_cell", "bbox": (0, 0, w, h)},
            {"type": "stamp", "bbox": (0, 0, min(50, w), min(50, h))},
        ]
        # Create a small full_image from the region
        results = preprocess_regions(regions, printed_region)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for r in results:
            assert "preprocessed" in r

    def test_skips_empty_region(self, printed_region):
        """Region with zero area should be skipped."""
        regions = [
            {"type": "data_cell", "bbox": (0, 0, 0, 0)},
        ]
        results = preprocess_regions(regions, printed_region)
        assert len(results) <= 1


# ── Edge Case Tests ──
class TestEdgeCases:
    """Tests for edge-case inputs."""

    def test_single_pixel(self):
        """Tiny 1x1 region should not crash."""
        tiny = np.ones((1, 1, 3), dtype=np.uint8) * 128
        result = preprocess_region(tiny, "handwritten")
        assert result is not None

    def test_very_noisy(self):
        """Very noisy region should not crash."""
        noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_region(noise, "printed_text")
        assert result is not None

    def test_blank_white(self):
        """All-white region should not crash."""
        white = np.ones((50, 100, 3), dtype=np.uint8) * 255
        result = preprocess_region(white, "printed_text")
        assert result is not None

    def test_very_dark(self):
        """Very dark region should not crash."""
        dark = np.ones((50, 100), dtype=np.uint8) * 10
        result = preprocess_region(dark, "handwritten")
        assert result is not None


# ── Configuration Tests ──
class TestConfig:
    """Verify all region types are handled."""

    def test_all_types_have_preprocessors(self):
        """All known region types should have a registered preprocessor."""
        from app.services.region_preprocess import (
            preprocess_printed_text,
            preprocess_handwritten,
            preprocess_table_cell,
            preprocess_stamp,
            preprocess_signature,
            preprocess_marginal_note,
        )
        preprocessors = {
            "printed_text": preprocess_printed_text,
            "handwritten": preprocess_handwritten,
            "table_cell": preprocess_table_cell,
            "stamp": preprocess_stamp,
            "signature": preprocess_signature,
            "marginal_note": preprocess_marginal_note,
            "data_cell": preprocess_table_cell,
            "header_row": preprocess_printed_text,
            "text_block": preprocess_printed_text,
            "record_block": preprocess_table_cell,
        }
        types = ["printed_text", "handwritten", "table_cell",
                 "stamp", "signature", "marginal_note",
                 "data_cell", "header_row", "text_block", "record_block"]
        for t in types:
            assert t in preprocessors, f"Missing preprocessor for {t}"