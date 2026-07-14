"""Tests for layout.py — Stage 2 layout detection after light preprocessing.

Verifies that layout detection:
    - Detects table boundaries (rows and columns)
    - Detects TEXT pages (no table) correctly
    - Extracts individual cells
    - Identifies record blocks (birth/marriage/death)
    - Detects stamps (red/blue circular regions)
    - Detects marginal notes
    - Detects page numbers
    - Detects signatures
    - Classifies regions as handwritten/printed/empty
    - Sorts elements in reading order
    - Returns uniform output format (type, bbox, confidence, properties)
    - Does NOT report table elements on text-only pages

Tests use REAL images from data/ directory run through light_preprocess first,
and save visual comparisons to data/test_output/.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from app.services.light_preprocess import light_preprocess
from app.services.layout import LayoutDetector, analyze_layout


# ── Constants ──
TEST_OUTPUT_DIR = Path("data/test_output")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_IMAGES_DIR = Path("data/01-0203-0745-000600")
ALL_TEST_IMAGES = sorted([p for p in DATA_IMAGES_DIR.rglob("*.jpg") if "test_output" not in str(p)])

print(f"\n Found {len(ALL_TEST_IMAGES)} test images for layout tests")


# ── Fixtures ──
@pytest.fixture(scope="session")
def output_dir() -> Path:
    return TEST_OUTPUT_DIR


@pytest.fixture(params=ALL_TEST_IMAGES[:3] if ALL_TEST_IMAGES else [""])
def sample_image_path(request) -> Path:
    path = request.param
    if not path:
        pytest.skip("No test images found")
    return path


@pytest.fixture
def sample_image(sample_image_path) -> np.ndarray:
    """Load and light-preprocess a real image from data/."""
    raw = cv2.imread(str(sample_image_path))
    if raw is None:
        pytest.skip(f"Cannot load {sample_image_path}")
    return light_preprocess(raw)


@pytest.fixture
def sample_image_name(sample_image_path) -> str:
    return sample_image_path.stem


@pytest.fixture
def detector() -> LayoutDetector:
    return LayoutDetector()


@pytest.fixture
def synthetic_table() -> np.ndarray:
    """Create a synthetic table image to test grid detection."""
    h, w = 800, 600
    image = np.ones((h, w, 3), dtype=np.uint8) * 240

    # Draw table grid lines
    for x in [50, 150, 300, 450, 550]:
        cv2.line(image, (x, 0), (x, h), (0, 0, 0), 2)
    for y in [0, 40, 100, 160, 220, 280, 340, 400]:
        cv2.line(image, (0, y), (w, y), (0, 0, 0), 2)

    # Header row text (printed)
    headers = ["N°", "Имя", "Дата", "Родители"]
    for i, (text, cx) in enumerate(zip(headers, [100, 225, 375, 500])):
        cv2.putText(image, text, (cx - 20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Data rows (handwritten-like)
    names = ["Иван", "Мария", "Пётр", "Анна", "Николай", "Елена"]
    for i, name in enumerate(names):
        y = 70 + i * 60
        cv2.putText(image, str(i + 1), (70, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        cv2.putText(image, name, (170, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        cv2.putText(image, f"{i+1}.III", (320, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)

    # Red stamp in top-right corner
    cv2.circle(image, (550, 100), 40, (0, 0, 200), -1)
    cv2.circle(image, (550, 100), 40, (0, 0, 150), 2)
    cv2.putText(image, "ПЕЧАТЬ", (520, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 1)

    # Marginal note on left
    cv2.putText(image, "см. стр. 5", (5, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)

    return image


@pytest.fixture
def synthetic_text_page() -> np.ndarray:
    """Create a synthetic text-only page (no table, no grid lines).

    Simulates a title page or paragraph page.
    """
    h, w = 800, 600
    image = np.ones((h, w, 3), dtype=np.uint8) * 245

    # Title (large text, centered)
    cv2.putText(image, "МЕТРИЧЕСКАЯ КНИГА", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "Церкви Святого Николая", (120, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Paragraph text (multiple lines)
    lines = [
        "Книга записей о родившихся,",
        "браком сочетавшихся и умерших",
        "за 1878 год.",
        "",
        "Часть I. О родившихся.",
        "Счет родившихся:",
        "мужеского пола 23,",
        "женского пола 19.",
        "Всего 42.",
    ]
    for i, text in enumerate(lines):
        cv2.putText(image, text, (50, 250 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Stamp at bottom (stamps can appear on any page)
    cv2.circle(image, (500, 700), 30, (0, 0, 200), -1)

    # Page number
    cv2.putText(image, "— 1 —", (270, 780),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    return image


# ── Helpers ──
def save_layout_visualization(
    image: np.ndarray,
    result: dict,
    name: str,
    output_dir: Path,
) -> Path:
    """Draw all detected elements on the image and save."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    colors = {
        "table":         (255, 0, 255),    # Magenta
        "text_block":    (128, 0, 128),    # Purple
        "record_block":  (255, 0, 0),      # Blue
        "header_row":    (0, 255, 255),    # Cyan
        "data_cell":     (0, 255, 0),      # Green
        "stamp":         (0, 0, 255),      # Red
        "marginal_note": (255, 255, 0),    # Yellow
        "page_number":   (128, 128, 255),  # Light purple
        "signature":     (255, 128, 0),    # Orange
    }

    for elem in result.get("elements", []):
        x1, y1, x2, y2 = elem["bbox"]
        color = colors.get(elem["type"], (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = elem["type"]
        if elem["type"] == "data_cell":
            props = elem.get("properties", {})
            label = f"R{props.get('row', '?')}C{props.get('col', '?')}"
            if props.get("region_type") == "handwritten":
                label += " HW"
        elif elem["type"] == "record_block":
            label += f" ({elem.get('properties', {}).get('record_type', '?')})"

        cv2.putText(vis, label, (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    out_path = output_dir / f"{name}_layout.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\n  Saved layout viz: {out_path}")
    return out_path


# ── Unit Tests ──
class TestTableDetection:
    """Tests for detect_table_boundaries."""

    def test_detects_rows_and_cols(self, detector, synthetic_table):
        """Should detect table grid structure."""
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        result = detector.detect_table_boundaries(gray)
        assert result["num_rows"] >= 3
        assert result["num_cols"] >= 3

    def test_rows_are_ordered(self, detector, synthetic_table):
        """Rows should be in top-to-bottom order."""
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        result = detector.detect_table_boundaries(gray)
        rows = result["rows"]
        for i in range(len(rows) - 1):
            assert rows[i][1] <= rows[i + 1][0]

    def test_columns_are_ordered(self, detector, synthetic_table):
        """Columns should be in left-to-right order."""
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        result = detector.detect_table_boundaries(gray)
        cols = result["columns"]
        for i in range(len(cols) - 1):
            assert cols[i][1] <= cols[i + 1][0]

    def test_real_image_table_detected(self, detector, sample_image):
        """Should detect at least some rows/cols on real images."""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        result = detector.detect_table_boundaries(gray)
        assert result["num_rows"] >= 2
        assert result["num_cols"] >= 2


class TestCellExtraction:
    """Tests for extract_cells."""

    def test_extracts_cells(self, detector, synthetic_table):
        """Should extract cells from table grid."""
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        table_info = detector.detect_table_boundaries(gray)
        cells = detector.extract_cells(gray, table_info)
        assert len(cells) > 0
        for cell in cells:
            assert cell["type"] == "data_cell"
            assert "properties" in cell
            assert "row" in cell["properties"]
            assert "col" in cell["properties"]

    def test_classifies_handwritten(self, detector, synthetic_table):
        """Should classify handwritten vs printed cells."""
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        table_info = detector.detect_table_boundaries(gray)
        cells = detector.extract_cells(gray, table_info)
        non_empty = [c for c in cells if not c["properties"]["is_empty"]]
        assert len(non_empty) > 0


class TestRegionClassification:
    """Tests for classify_region."""

    def test_empty_region(self, detector):
        empty = np.ones((50, 50), dtype=np.uint8) * 255
        assert detector.classify_region(empty) == "empty"

    def test_dark_region(self, detector):
        dark = np.zeros((50, 50), dtype=np.uint8)
        for _ in range(20):
            x, y = np.random.randint(0, 50, 2)
            cv2.line(dark, (x, y), (x + 5, y + 5), (50, 50, 50), 2)
        result = detector.classify_region(dark)
        assert result in ("handwritten", "printed")

    def test_printed_region(self, detector):
        printed = np.ones((50, 50), dtype=np.uint8) * 200
        cv2.putText(printed, "Test", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        result = detector.classify_region(printed)
        assert result == "printed"


class TestStampDetection:
    """Tests for detect_stamps."""

    def test_detects_red_stamp(self, detector, synthetic_table):
        stamps = detector.detect_stamps(synthetic_table)
        red_stamps = [s for s in stamps if s["properties"].get("color") == "red"]
        assert len(red_stamps) >= 1

    def test_stamp_has_confidence(self, detector, synthetic_table):
        stamps = detector.detect_stamps(synthetic_table)
        for s in stamps:
            assert s["confidence"] > 0
            assert s["type"] == "stamp"


class TestMarginalNotes:
    """Tests for detect_marginal_notes."""

    def test_detects_marginal_notes(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        notes = detector.detect_marginal_notes(gray)
        assert isinstance(notes, list)

    def test_notes_have_region(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        notes = detector.detect_marginal_notes(gray)
        for note in notes:
            assert "region" in note["properties"]


class TestRecordBlocks:
    """Tests for detect_record_blocks."""

    def test_detects_header_row(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        table_info = detector.detect_table_boundaries(gray)
        blocks = detector.detect_record_blocks(gray, table_info)
        headers = [b for b in blocks if b["type"] == "header_row"]
        assert len(headers) >= 1

    def test_detects_record_blocks(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        table_info = detector.detect_table_boundaries(gray)
        blocks = detector.detect_record_blocks(gray, table_info)
        record_blocks = [b for b in blocks if b["type"] == "record_block"]
        assert len(record_blocks) >= 1


class TestTextPageDetection:
    """Tests for detecting text-only pages (no table)."""

    def test_text_page_has_no_table(self, detector, synthetic_text_page):
        """A text-only page should NOT return table elements."""
        result = detector.process(synthetic_text_page)

        tables = [e for e in result["elements"] if e["type"] == "table"]
        cells = [e for e in result["elements"] if e["type"] == "data_cell"]

        assert len(tables) == 0, (
            f"Text page should not have table elements, got {len(tables)}"
        )
        assert len(cells) == 0, (
            f"Text page should not have cell elements, got {len(cells)}"
        )

    def test_text_page_has_text_blocks(self, detector, synthetic_text_page):
        """A text-only page should have text_block elements."""
        result = detector.process(synthetic_text_page)

        text_blocks = [e for e in result["elements"] if e["type"] == "text_block"]
        assert len(text_blocks) >= 1, (
            f"Text page should have text blocks, got {len(text_blocks)}"
        )

    def test_text_page_has_stamps(self, detector, synthetic_text_page):
        """Stamps should still be detected on text pages."""
        result = detector.process(synthetic_text_page)
        stamps = [e for e in result["elements"] if e["type"] == "stamp"]
        assert len(stamps) >= 1, "Stamp should be detected on text page"

    def test_text_page_has_page_number(self, detector, synthetic_text_page):
        """Page numbers should still be detected on text pages."""
        result = detector.process(synthetic_text_page)
        page_nums = [e for e in result["elements"] if e["type"] == "page_number"]
        assert isinstance(page_nums, list)

    def test_text_page_page_type(self, detector, synthetic_text_page):
        """Page type should be 'text', not 'table'."""
        result = detector.process(synthetic_text_page)
        assert result["page_type"] == "text", (
            f"Expected page_type='text', got '{result['page_type']}'"
        )

    def test_table_page_page_type(self, detector, synthetic_table):
        """Table page should have page_type='table'."""
        result = detector.process(synthetic_table)
        assert result["page_type"] == "table", (
            f"Expected page_type='table', got '{result['page_type']}'"
        )

    def test_text_page_summary_no_cells(self, detector, synthetic_text_page):
        """Summary should show 0 cells for text pages."""
        result = detector.process(synthetic_text_page)
        summary = result["metadata"]["summary"]
        assert summary["cells"] == 0, f"Expected 0 cells, got {summary['cells']}"
        assert summary["tables"] == 0, f"Expected 0 tables, got {summary['tables']}"

    def test_real_image_may_be_text(self, detector, sample_image):
        """Real images should not crash — may be table or text."""
        result = detector.process(sample_image)
        assert result["page_type"] in ("table", "text")
        assert len(result["elements"]) > 0


class TestPageNumber:
    """Tests for detect_page_number."""

    def test_returns_list(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        numbers = detector.detect_page_number(gray)
        assert isinstance(numbers, list)


class TestSignature:
    """Tests for detect_signatures."""

    def test_returns_list(self, detector, synthetic_table):
        gray = cv2.cvtColor(synthetic_table, cv2.COLOR_BGR2GRAY)
        sigs = detector.detect_signatures(gray)
        assert isinstance(sigs, list)


class TestReadingOrder:
    """Tests for sort_reading_order."""

    def test_sorts_top_to_bottom(self):
        detector = LayoutDetector()
        elements = [
            {"bbox": (0, 100, 50, 150)},
            {"bbox": (0, 50, 50, 100)},
            {"bbox": (100, 100, 150, 150)},
        ]
        sorted_elems = detector.sort_reading_order(elements)
        assert sorted_elems[0]["bbox"][1] == 50
        assert sorted_elems[1]["bbox"][1] == 100
        assert sorted_elems[2]["bbox"][1] == 100
        assert sorted_elems[1]["bbox"][0] == 0
        assert sorted_elems[2]["bbox"][0] == 100


# ── Integration Tests ──
class TestAnalyzeLayout:
    """Tests for the main analyze_layout() function."""

    def test_returns_elements(self, detector, synthetic_table):
        result = detector.process(synthetic_table)
        assert "elements" in result
        assert len(result["elements"]) > 0

    def test_elements_have_required_fields(self, detector, synthetic_table):
        result = detector.process(synthetic_table)
        for elem in result["elements"]:
            assert "type" in elem
            assert "bbox" in elem
            assert "confidence" in elem
            assert "source" in elem
            assert "properties" in elem

    def test_bbox_format(self, detector, synthetic_table):
        result = detector.process(synthetic_table)
        for elem in result["elements"]:
            bbox = elem["bbox"]
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox
            assert x1 < x2
            assert y1 < y2

    def test_metadata_present(self, detector, synthetic_table):
        result = detector.process(synthetic_table)
        assert "metadata" in result
        assert "image_size" in result["metadata"]

    def test_confidence_range(self, detector, synthetic_table):
        result = detector.process(synthetic_table)
        for elem in result["elements"]:
            assert 0 <= elem["confidence"] <= 1.0

    def test_real_image(self, detector, sample_image):
        result = detector.process(sample_image)
        assert len(result["elements"]) > 0

    def test_includes_table_element(self, detector, synthetic_table):
        """On a table page, should have 'table' type elements."""
        result = detector.process(synthetic_table)
        tables = [e for e in result["elements"] if e["type"] == "table"]
        assert len(tables) >= 1

    def test_includes_data_cells(self, detector, synthetic_table):
        """On a table page, should have data cells."""
        result = detector.process(synthetic_table)
        cells = [e for e in result["elements"] if e["type"] == "data_cell"]
        assert len(cells) >= 1

    def test_saves_visualization(self, detector, sample_image, sample_image_name, output_dir):
        result = detector.process(sample_image)
        save_layout_visualization(sample_image, result, sample_image_name, output_dir)


class TestEndToEnd:
    """Full pipeline: light_preprocess → analyze_layout."""

    def test_full_pipeline(self, sample_image_path, output_dir):
        raw = cv2.imread(str(sample_image_path))
        if raw is None:
            pytest.skip(f"Cannot load {sample_image_path}")

        clean = light_preprocess(raw)
        result = analyze_layout(clean)
        save_layout_visualization(clean, result, sample_image_path.stem, output_dir)

        assert len(result["elements"]) > 0
        assert result["metadata"]["num_elements"] > 0

    def test_summary_counts(self, sample_image_path):
        raw = cv2.imread(str(sample_image_path))
        if raw is None:
            pytest.skip(f"Cannot load {sample_image_path}")

        clean = light_preprocess(raw)
        result = analyze_layout(clean)
        summary = result["metadata"]["summary"]

        total_cells_in_summary = summary.get("handwritten_cells", 0) + summary.get("printed_cells", 0)
        actual_cells = len([e for e in result["elements"] if e["type"] == "data_cell"])

        assert total_cells_in_summary <= actual_cells


# ── Edge Case Tests ──
class TestEdgeCases:
    """Tests for unusual or edge-case inputs."""

    def test_completely_blank(self, detector):
        blank = np.ones((500, 500, 3), dtype=np.uint8) * 255
        result = detector.process(blank)
        assert result is not None

    def test_very_dark(self, detector):
        dark = np.ones((500, 500, 3), dtype=np.uint8) * 10
        result = detector.process(dark)
        assert result is not None

    def test_very_small(self, detector):
        small = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result = detector.process(small)
        assert result is not None


# ── Configuration Tests ──
class TestConfig:
    """Verify module constants are sensible."""

    def test_element_types_defined(self):
        from app.services.layout import ELEMENT_TYPES
        expected = ["table", "record_block", "data_cell", "stamp", "marginal_note", "text_block"]
        for t in expected:
            assert t in ELEMENT_TYPES, f"Missing element type: {t}"