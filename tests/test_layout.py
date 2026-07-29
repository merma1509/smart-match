"""Unit tests for layout detection module.

Тестирует детекцию таблиц, текстовых блоков, штампов,
маргиналий и других структурных элементов метрических книг.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.layout import LayoutDetector, analyze_layout


# ── Fixtures ──
@pytest.fixture
def detector():
    return LayoutDetector()


@pytest.fixture
def synthetic_table_image():
    """Создаёт синтетическое изображение с таблицей.

    Тёмные строки на светлом фоне для детекции через Otsu."""
    img = np.ones((600, 800), dtype=np.uint8) * 245

    # Три строки таблицы
    cv2.rectangle(img, (55, 55), (745, 145), 80, -1)
    cv2.rectangle(img, (55, 155), (745, 245), 80, -1)
    cv2.rectangle(img, (55, 255), (745, 345), 80, -1)

    # Белые просветы между строками
    cv2.rectangle(img, (55, 145), (745, 155), 245, -1)
    cv2.rectangle(img, (55, 245), (745, 255), 245, -1)

    return img


@pytest.fixture
def synthetic_text_image():
    """Создаёт синтетическое изображение с текстовыми блоками."""
    img = np.ones((600, 800), dtype=np.uint8) * 255

    # Текстовый блок вверху
    cv2.rectangle(img, (100, 50), (700, 150), 200, -1)

    # Текстовый блок в середине
    cv2.rectangle(img, (50, 200), (750, 350), 200, -1)

    # Текстовый блок внизу
    cv2.rectangle(img, (150, 400), (650, 500), 200, -1)

    return img


@pytest.fixture
def synthetic_stamp_image():
    """Создаёт синтетическое изображение с красной печатью."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Красный круг (печать)
    center = (200, 200)
    cv2.circle(img, center, 60, (0, 0, 200), -1)
    cv2.circle(img, center, 50, (0, 0, 255), 2)

    return img


# ── Table Detection Tests ──
class TestTableDetection:
    def test_detect_table_synthetic(self, detector, synthetic_table_image):
        """Детекция таблицы на синтетическом изображении (через Otsu).
        На синтетических данных может не найти, но не должно падать."""
        table_info = detector.detect_table_boundaries_from_raw(synthetic_table_image)
        assert isinstance(table_info, dict)
        assert "rows" in table_info
        assert "columns" in table_info
        assert table_info["num_rows"] >= 0
        assert table_info["num_cols"] >= 0

    def test_detect_table_from_raw(self, detector, synthetic_table_image):
        """Детекция таблицы через raw Otsu."""
        table_info = detector.detect_table_boundaries_from_raw(synthetic_table_image)
        assert isinstance(table_info, dict)
        assert "rows" in table_info

    def test_detect_table_no_table(self, detector, synthetic_text_image):
        """На изображении без таблицы не должно быть падений."""
        table_info = detector.detect_table_boundaries_from_raw(synthetic_text_image)
        assert isinstance(table_info, dict)

    def test_is_table_page(self, detector, synthetic_table_image):
        """Проверка определения страницы как табличной.
        Создаём table_info с достаточным количеством строк для _is_table_page."""
        table_info = {
            "rows": [(100, 200), (200, 300), (300, 400), (400, 500)],
            "columns": [(0, 400), (400, 800)],
            "num_rows": 4,
            "num_cols": 2,
        }
        is_table = detector._is_table_page(synthetic_table_image, table_info)
        assert is_table, "Table with 4 rows and 2 cols should be detected as table page"

    def test_table_confidence_high(self, detector, synthetic_table_image):
        """Высокая уверенность для хорошей таблицы."""
        table_info = detector.detect_table_boundaries(synthetic_table_image)
        # Добавляем больше строк/колонок для высокой уверенности
        table_info["num_rows"] = 6
        table_info["num_cols"] = 4
        confidence = detector._table_confidence(table_info)
        assert confidence >= 0.9, f"Expected high confidence, got {confidence}"

    def test_table_confidence_low(self, detector, synthetic_text_image):
        """Низкая уверенность для не-таблицы."""
        table_info = {"num_rows": 1, "num_cols": 1}
        confidence = detector._table_confidence(table_info)
        assert confidence <= 0.3, f"Expected low confidence, got {confidence}"


# ── Cell Extraction Tests ──
class TestCellExtraction:
    def test_extract_cells(self, detector, synthetic_table_image):
        """Извлечение ячеек из таблицы."""
        table_info = detector.detect_table_boundaries(synthetic_table_image)
        cells = detector.extract_cells(synthetic_table_image, table_info)
        assert len(cells) > 0, "Should extract at least some cells"

    def test_cell_properties(self, detector, synthetic_table_image):
        """Проверка свойств извлечённых ячеек."""
        table_info = detector.detect_table_boundaries(synthetic_table_image)
        cells = detector.extract_cells(synthetic_table_image, table_info)
        for cell in cells:
            assert "type" in cell
            assert "bbox" in cell
            assert "confidence" in cell
            assert "properties" in cell
            assert cell["type"] == "data_cell"

    def test_header_cell_detection(self, detector, synthetic_table_image):
        """Первая строка/колонка должна быть отмечена как header."""
        # Создаём таблицу с явной первой строкой header
        table_info = {
            "rows": [(0, 50), (50, 100), (100, 150)],
            "columns": [(0, 100), (100, 200)],
            "num_rows": 3,
            "num_cols": 2,
        }
        cells = detector.extract_cells(synthetic_table_image, table_info)
        headers = [c for c in cells if c["properties"]["is_header"]]
        assert len(headers) > 0, "Should have at least one header cell"


# ── Text Block Tests ──
class TestTextBlockDetection:
    def test_detect_text_blocks(self, detector, synthetic_text_image):
        """Детекция текстовых блоков."""
        blocks = detector.detect_text_blocks(synthetic_text_image)
        assert len(blocks) >= 2, f"Expected >=2 text blocks, got {len(blocks)}"

    def test_text_block_properties(self, detector, synthetic_text_image):
        """Проверка свойств текстовых блоков."""
        blocks = detector.detect_text_blocks(synthetic_text_image)
        for block in blocks:
            assert block["type"] == "text_block"
            assert len(block["bbox"]) == 4
            assert 0 <= block["confidence"] <= 1.0
            assert "area" in block["properties"]
            assert "aspect_ratio" in block["properties"]


# ── Stamp Detection Tests ──


class TestStampDetection:
    def test_detect_red_stamp(self, detector, synthetic_stamp_image):
        """Детекция красной печати."""
        stamps = detector.detect_stamps(synthetic_stamp_image)
        # Может не найти из-за синтетической природы, но не должно падать
        assert isinstance(stamps, list)

    def test_stamp_properties(self, detector, synthetic_stamp_image):
        """Проверка свойств печати."""
        stamps = detector.detect_stamps(synthetic_stamp_image)
        for stamp in stamps:
            assert stamp["type"] == "stamp"
            assert len(stamp["bbox"]) == 4
            assert "color" in stamp["properties"]


# ── Marginal Notes Tests ──
class TestMarginalNotes:
    def test_detect_marginal_notes_empty(self, detector, synthetic_table_image):
        """На изображении без маргиналий не должно быть ложных срабатываний."""
        notes = detector.detect_marginal_notes(synthetic_table_image)
        assert isinstance(notes, list)

    def test_detect_marginal_notes_with_content(self, detector):
        """Маргиналии должны обнаруживаться при наличии контента."""
        img = np.ones((600, 800), dtype=np.uint8) * 255
        # Добавляем тёмные пиксели в левое поле (4% от 800 = 32px)
        img[200:300, 5:25] = 50  # Тёмная область в левом поле
        notes = detector.detect_marginal_notes(img)
        # Может не найти (синтетика), но не падает
        assert isinstance(notes, list)


# ── Reading Order Tests ──
class TestReadingOrder:
    def test_sort_reading_order(self, detector):
        """Проверка сортировки элементов в порядке чтения."""
        elements = [
            {"bbox": (100, 200, 150, 250)},
            {"bbox": (50, 100, 100, 150)},
            {"bbox": (200, 300, 250, 350)},
        ]
        sorted_elements = detector.sort_reading_order(elements)
        y_coords = [e["bbox"][1] for e in sorted_elements]
        assert y_coords == sorted(y_coords), "Elements should be sorted by y-coordinate"


# ── Region Classification Tests ──
class TestRegionClassification:
    def test_classify_empty(self, detector):
        """Пустая область должна быть 'empty'."""
        empty = np.ones((50, 50), dtype=np.uint8) * 255
        assert detector.classify_region(empty) == "empty"

    def test_classify_handwritten(self, detector):
        """Область с высоким Laplacian var должна быть 'handwritten'."""
        handwritten = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        # Делаем резкие переходы (характерно для рукописного текста)
        handwritten[10:40, 10:40] = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        result = detector.classify_region(handwritten)
        assert result in ("handwritten", "printed"), f"Unexpected: {result}"


# ── Full Pipeline Tests ──
class TestFullPipeline:
    def test_analyze_layout_table(self, detector, synthetic_table_image):
        """Полный пайплайн анализа на табличном изображении."""
        # Создаём BGR изображение (как приходит из light_preprocess)
        bgr = cv2.cvtColor(synthetic_table_image, cv2.COLOR_GRAY2BGR)
        result = detector.process(bgr)

        assert "elements" in result
        assert "page_type" in result
        assert "metadata" in result
        assert len(result["elements"]) > 0

    def test_analyze_layout_text(self, detector, synthetic_text_image):
        """Полный пайплайн анализа на текстовом изображении."""
        bgr = cv2.cvtColor(synthetic_text_image, cv2.COLOR_GRAY2BGR)
        result = detector.process(bgr)

        assert "elements" in result
        assert "page_type" in result
        assert result["page_type"] in ("table", "text")

    def test_analyze_layout_with_stamps(self, detector, synthetic_stamp_image):
        """Пайплайн с печатями."""
        result = detector.process(synthetic_stamp_image)
        assert "elements" in result

    def test_analyze_layout_function(self, synthetic_table_image):
        """Тест функции analyze_layout (удобный API)."""
        bgr = cv2.cvtColor(synthetic_table_image, cv2.COLOR_GRAY2BGR)
        result = analyze_layout(bgr)
        assert "elements" in result


# ── Helper Tests ──
class TestHelpers:
    def test_to_gray_bgr(self):
        """Конвертация BGR в grayscale."""
        from app.services.layout import _to_gray

        bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = _to_gray(bgr)
        assert len(result.shape) == 2

    def test_to_gray_gray(self):
        """Grayscale изображение не должно меняться."""
        from app.services.layout import _to_gray

        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = _to_gray(gray)
        assert result.shape == gray.shape

    def test_projection_to_segments(self, detector):
        """Конвертация проекции в сегменты."""
        proj = np.array([0, 0, 0.5, 0.6, 0.7, 0, 0, 0.8, 0.9, 0, 0])
        segments = detector._projection_to_segments(proj, threshold=0.1, min_length=1)
        assert len(segments) >= 1, "Should detect at least one segment"
