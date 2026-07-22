"""Layout detection for Russian historical metrical books.
Takes light-preprocessed images (Stage 1 output) and detects:
    - Table boundaries and grid structure
    - Record blocks (birth/marriage/death)
    - Individual cells (handwritten vs printed)
    - Marginal notes
    - Stamps and seals
    - Page numbers
    - Signatures

Output: list of elements with type, bbox, confidence, and properties.

References:
    [1] Existing layout.py heuristic methods
    [2] OpenCV Hough line transform docs
    [3] HSV color segmentation for stamp detection
"""


import cv2
import numpy as np
from loguru import logger

# ── Constants ──
ELEMENT_TYPES = [
    "table",
    "record_block",
    "header_row",
    "data_row",
    "data_cell",
    "text_block",
    "marginal_note",
    "stamp",
    "signature",
    "page_number",
    "decorative",
    "empty",
]

TABLE_LINE_THRESHOLD = 200  # HoughLines threshold
MIN_ROW_HEIGHT = 20  # Minimum row height in pixels
MIN_COL_WIDTH = 30  # Minimum column width in pixels
PROJECTION_THRESHOLD = 0.08  # Row detection sensitivity


# ── Helper ──


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ── Detection Modules ──
class LayoutDetector:
    """Detects structural elements in Russian metrical book pages.

    Input: light-preprocessed BGR image (~2000px max dim).
    Output: dict with 'elements' list and 'metadata'.
    """

    def __init__(self):
        self.model_path = "app/models/layout/yolov8n.pt"  # For future fine-tuned model
        self.yolo_model = None  # Will be loaded lazily if needed

    # ── 1. Table Boundary Detection ──
    def detect_table_boundaries(self, gray: np.ndarray) -> dict:
        """Detect table grid using morphological line detection on edges.

        Uses Canny edge detection + morphological filtering to find
        long horizontal and vertical lines that form table boundaries"""
        h, w = gray.shape

        # Step 1: Edge detection with low thresholds to catch faint lines
        edges = cv2.Canny(gray, 20, 80)

        # Step 2: Morphological opening to extract ONLY long horizontal lines
        # A 50px kernel means: keep only horizontal segments >= 50px long
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)

        # Step 3: Same for vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

        # Step 4: Horizontal projection of horizontal lines only
        h_proj = np.sum(h_lines > 0, axis=1).astype(float)
        h_proj = h_proj / max(np.max(h_proj), 1)

        # Step 5: Vertical projection of vertical lines only
        v_proj = np.sum(v_lines > 0, axis=0).astype(float)
        v_proj = v_proj / max(np.max(v_proj), 1)

        # Step 6: Detect rows from horizontal projection
        rows = self._projection_to_segments(h_proj, threshold=0.1, min_length=15)

        # Step 7: Detect columns from vertical projection
        cols = self._projection_to_segments(v_proj, threshold=0.05, min_length=30)

        # Step 8: Fallback — if too few rows/cols, try line intersection detection
        if len(rows) < 3:
            rows = self._detect_rows_from_lines(h_lines, gray.shape)

        if len(cols) < 2:
            cols = self._detect_cols_from_lines(v_lines, gray.shape)

        return {
            "rows": rows,
            "columns": cols,
            "num_rows": len(rows),
            "num_cols": len(cols),
        }

    def detect_table_boundaries_from_raw(self, gray_raw: np.ndarray) -> dict:
        """Detect table grid using Otsu on RAW (non-illumination-normalized) gray.

        Uses the original grayscale BEFORE illumination normalization,
        because that step destroys thin table lines.
        """
        h, w = gray_raw.shape

        # Otsu thresholding on the raw image — table lines are still visible
        _, binary = cv2.threshold(gray_raw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological open to extract horizontal lines (≥50px long)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Morphological open to extract vertical lines (≥50px tall)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Horizontal projection
        h_proj = np.sum(h_lines > 0, axis=1) / 255.0
        h_proj = h_proj / max(np.max(h_proj), 1)

        # Vertical projection
        v_proj = np.sum(v_lines > 0, axis=0) / 255.0
        v_proj = v_proj / max(np.max(v_proj), 1)

        # Detect rows
        rows = self._projection_to_segments(h_proj, threshold=0.08, min_length=15)

        # Detect columns
        cols = self._projection_to_segments(v_proj, threshold=0.05, min_length=30)

        # Fallback to line-based detection
        if len(rows) < 3:
            rows = self._detect_rows_from_lines(h_lines, gray_raw.shape)
        if len(cols) < 2:
            cols = self._detect_cols_from_lines(v_lines, gray_raw.shape)

        return {
            "rows": rows,
            "columns": cols,
            "num_rows": len(rows),
            "num_cols": len(cols),
        }

    def detect_table_robust(self, image: np.ndarray, gray_raw: np.ndarray = None) -> dict:
        """ROBUST table detection — works on original image or raw grayscale.

        Uses multiple strategies and picks the best result:
        1. Try morphological on raw gray (if available)
        2. Try HSV color filtering (tables often have printed blue/purple headers)
        3. Try adaptive thresholding
        4. Fallback: treat whole page as one big table with 2 columns

        Returns unified table_info dict.
        """
        # Strategy 1: Use raw gray if available
        if gray_raw is not None:
            result = self.detect_table_boundaries_from_raw(gray_raw)
            if result["num_rows"] >= 2 and result["num_cols"] >= 2:
                logger.info(
                    f"Table found via raw Otsu: {result['num_rows']}r x {result['num_cols']}c"
                )
                return result

        # Strategy 2: Try on preprocessed gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Try with different Canny thresholds
        for low, high in [(20, 80), (30, 100), (50, 150)]:
            edges = cv2.Canny(gray, low, high)

            # Horizontal lines
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
            h_proj = np.sum(h_lines > 0, axis=1).astype(float)
            h_proj = h_proj / max(np.max(h_proj), 1)

            # Vertical lines
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
            v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
            v_proj = np.sum(v_lines > 0, axis=0).astype(float)
            v_proj = v_proj / max(np.max(v_proj), 1)

            # Detect rows and columns with lower thresholds
            rows = self._projection_to_segments(h_proj, threshold=0.05, min_length=10)
            cols = self._projection_to_segments(v_proj, threshold=0.03, min_length=20)

            if len(rows) >= 2 and len(cols) >= 2:
                logger.info(f"Table found via Canny({low},{high}): {len(rows)}r x {len(cols)}c")
                return {"rows": rows, "columns": cols, "num_rows": len(rows), "num_cols": len(cols)}

        # Strategy 3: Force table detection — split page into logical regions
        h, w = gray.shape

        # Use projection profiles to find text lines (NOT table lines)
        # In metrical books, text is organized in columns even without visible lines
        bin_inv = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(bin_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Horizontal projection of text
        h_proj = np.sum(binary > 0, axis=1).astype(float)
        h_proj = h_proj / max(np.max(h_proj), 1)

        # Find rows where text exists
        text_rows = []
        in_row = False
        start = 0
        for i in range(len(h_proj)):
            if h_proj[i] > 0.005 and not in_row:  # Very low threshold
                start = i
                in_row = True
            elif h_proj[i] <= 0.005 and in_row:
                if i - start > 5:  # At least 5px tall
                    text_rows.append((start, i))
                in_row = False
        if in_row and len(h_proj) - start > 5:
            text_rows.append((start, len(h_proj)))

        # Also try to find column-like structures via vertical projection
        v_proj = np.sum(binary > 0, axis=0).astype(float)
        v_proj = v_proj / max(np.max(v_proj), 1)

        text_cols = []
        in_col = False
        start = 0
        for i in range(len(v_proj)):
            if v_proj[i] > 0.003 and not in_col:
                start = i
                in_col = True
            elif v_proj[i] <= 0.003 and in_col:
                if i - start > 30:  # At least 30px wide
                    text_cols.append((start, i))
                in_col = False
        if in_col and len(v_proj) - start > 30:
            text_cols.append((start, len(v_proj)))

        if len(text_rows) >= 3 and len(text_cols) >= 1:
            logger.info(f"Table forced via text projection: {len(text_rows)}r x {len(text_cols)}c")
            return {
                "rows": text_rows,
                "columns": text_cols if len(text_cols) >= 2 else [(0, w)],
                "num_rows": len(text_rows),
                "num_cols": max(len(text_cols), 1),
            }

        # Strategy 4: Ultimate fallback — treat as single-column table
        logger.info("No table found, using page as single region")
        return {
            "rows": text_rows if len(text_rows) >= 2 else [(0, h)],
            "columns": [(0, w)],
            "num_rows": max(len(text_rows), 1),
            "num_cols": 1,
        }

    def _projection_to_segments(self, proj: np.ndarray, threshold: float, min_length: int) -> list:
        """Convert projection profile to list of (start, end) segments."""
        segments = []
        in_segment = False
        start = 0
        for i in range(len(proj)):
            if proj[i] > threshold and not in_segment:
                start = i
                in_segment = True
            elif proj[i] <= threshold and in_segment:
                if i - start > min_length:
                    segments.append((start, i))
                in_segment = False
        if in_segment and len(proj) - start > min_length:
            segments.append((start, len(proj)))
        return segments

    def _detect_rows_from_lines(self, h_lines: np.ndarray, shape: tuple) -> list:
        """Fallback: detect row positions from horizontal line segments."""
        h_sum = np.sum(h_lines > 0, axis=1)
        h_th = shape[1] * 0.25
        return self._detect_line_positions(h_sum, h_th, min_gap=15, shape_max=shape[0])

    def _detect_cols_from_lines(self, v_lines: np.ndarray, shape: tuple) -> list:
        """Fallback: detect column positions from vertical line segments."""
        v_sum = np.sum(v_lines > 0, axis=0)
        v_th = shape[0] * 0.25
        return self._detect_line_positions(v_sum, v_th, min_gap=30, shape_max=shape[1])

    def _detect_line_positions(
        self, profile: np.ndarray, threshold: float, min_gap: int, shape_max: int
    ) -> list:
        """Generic line position detection from projection profile."""
        positions = []
        in_line = False
        start = 0
        for i in range(len(profile)):
            if profile[i] > threshold and not in_line:
                start = i
                in_line = True
            elif profile[i] <= threshold and in_line:
                if i - start > 2:
                    positions.append((start + i) // 2)
                in_line = False
        segments = []
        prev = 0
        for pos in positions:
            if pos - prev > min_gap:
                segments.append((prev, pos))
            prev = pos
        if prev < shape_max:
            segments.append((prev, shape_max))
        return segments

    # ── 2. Table Page Detection ──
    def _is_table_page(self, gray: np.ndarray, table_info: dict) -> bool:
        """Relaxed table page detection for metrical books."""
        h, w = gray.shape

        # 1. Check row/column count — relaxed
        if table_info["num_rows"] < 2:
            return False

        # 2. Check if rows cover a significant portion — relaxed to 20%
        rows = table_info["rows"]
        total_row_area = sum(y2 - y1 for y1, y2 in rows)
        row_coverage = total_row_area / h if h > 0 else 0
        if row_coverage < 0.2:
            return False

        # 3. Line density check — only if we have enough rows
        if table_info["num_rows"] >= 3 and table_info["num_cols"] >= 1:
            return True

        # 4. For pages with at least 2 rows — assume it's tabular if rows are regular
        if table_info["num_rows"] >= 4:
            return True

        # 5. Default: if we have rows and at least some structure
        if table_info["num_rows"] >= 2:
            return True

        return False

    def _table_confidence(self, table_info: dict) -> float:
        """Calculate confidence that this is actually a table page."""
        rows = table_info["num_rows"]
        cols = table_info["num_cols"]

        if rows >= 5 and cols >= 3:
            return 0.9  # Strong table
        elif rows >= 3 and cols >= 2:
            return 0.7  # Moderate table
        elif rows >= 2 and cols >= 2:
            return 0.5  # Weak table
        else:
            return 0.3  # Probably not a table

    # ── 3. Cell Extraction ──
    def extract_cells(self, gray: np.ndarray, table_info: dict) -> list:
        """Extract individual cells from table grid."""
        cells = []
        rows = table_info.get("rows", [])
        cols = table_info.get("columns", [])

        for ri, (ry1, ry2) in enumerate(rows):
            for ci, (cx1, cx2) in enumerate(cols):
                cell_img = gray[ry1:ry2, cx1:cx2]
                if cell_img.size == 0:
                    continue
                is_empty = np.mean(cell_img) > 250
                cell_type = self.classify_region(cell_img)
                cells.append(
                    {
                        "type": "data_cell",
                        "bbox": (cx1, ry1, cx2, ry2),
                        "confidence": 0.0 if is_empty else (0.8 if cell_type != "empty" else 0.3),
                        "source": "table_grid",
                        "properties": {
                            "row": ri,
                            "col": ci,
                            "is_header": ri == 0 or ci == 0,
                            "region_type": cell_type,
                            "is_empty": is_empty,
                        },
                    }
                )
        return cells

    # ── 4. Text Block Detection (for non-table pages) ──
    def detect_text_blocks(self, gray: np.ndarray) -> list:
        """Detect paragraph text blocks on non-table pages.

        Uses contour analysis to find text regions.
        """
        h, w = gray.shape
        text_blocks = []

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to merge nearby text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Too small
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0

            # Text blocks are typically wider than tall
            if 0.5 < aspect < 10 and ch > 15:
                text_blocks.append(
                    {
                        "type": "text_block",
                        "bbox": (x, y, x + cw, y + ch),
                        "confidence": min(0.8, area / (h * w) * 10),
                        "source": "contour_analysis",
                        "properties": {
                            "area": int(area),
                            "aspect_ratio": round(aspect, 2),
                        },
                    }
                )

        return text_blocks

    # ── 5. Record Block Detection ──
    def detect_record_blocks(self, gray: np.ndarray, table_info: dict) -> list:
        """Cluster rows into record blocks (birth/marriage/death).

        Adapted from [1] detect_record_blocks().
        """
        blocks = []
        rows = table_info.get("rows", [])
        if not rows:
            return blocks

        current_block = []
        prev_height = 0
        header_done = False

        for i, (y1, y2) in enumerate(rows):
            height = y2 - y1

            # First row is typically the header
            if not header_done:
                blocks.append(
                    {
                        "type": "header_row",
                        "bbox": (0, y1, gray.shape[1], y2),
                        "confidence": 0.9,
                        "source": "position",
                        "properties": {"row_index": i, "is_header": True},
                    }
                )
                header_done = True
                prev_height = height
                continue

            # Gap or height change indicates new block
            if prev_height > 0 and abs(height - prev_height) > 15:
                self._flush_block(current_block, gray, blocks)
                current_block = []

            current_block.append((y1, y2))
            prev_height = height

        self._flush_block(current_block, gray, blocks)

        # Assign record types
        blocks = self.assign_record_type(gray, blocks)

        return blocks

    def _flush_block(self, current_block: list, gray: np.ndarray, blocks: list):
        """Flush accumulated rows as a record block."""
        if not current_block:
            return
        block_y1 = current_block[0][0]
        block_y2 = current_block[-1][1]
        blocks.append(
            {
                "type": "record_block",
                "bbox": (0, block_y1, gray.shape[1], block_y2),
                "confidence": 0.8,
                "source": "row_clustering",
                "properties": {
                    "row_count": len(current_block),
                    "record_type": "unknown",
                },
            }
        )

    # ── 6. Region Classification ──
    def classify_region(self, cell_img: np.ndarray) -> str:
        """Classify region as 'printed', 'handwritten', or 'empty'.

        Adapted from [1] classify_region().
        """
        if cell_img.size == 0 or np.mean(cell_img) > 250:
            return "empty"

        local_var = cv2.Laplacian(cell_img, cv2.CV_64F).var()
        edges = cv2.Canny(cell_img, 50, 150)
        edge_density = np.mean(edges > 0)

        if local_var > 300 and edge_density > 0.08:
            return "handwritten"
        elif local_var > 50:
            return "printed"
        else:
            return "empty"

    # ── 7. Stamp Detection ──
    def detect_stamps(self, image: np.ndarray) -> list:
        """Detect red/blue circular stamps using HSV color segmentation.

        References: [2] OpenCV HSV color segmentation, HoughCircles.
        """
        stamps = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # Red color masks (red wraps around hue 0/180)
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Blue color mask
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))

        for color_name, mask in [("red", red_mask), ("blue", blue_mask)]:
            # Clean mask
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 400 or area > w * h * 0.2:
                    continue

                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                x, y, cw, ch = cv2.boundingRect(cnt)
                if circularity > 0.5 or (0.7 < cw / ch < 1.3 and cw > 20):
                    stamps.append(
                        {
                            "type": "stamp",
                            "bbox": (x, y, x + cw, y + ch),
                            "confidence": round(min(0.9, 0.4 + circularity / 2), 2),
                            "source": "color_detection",
                            "properties": {
                                "color": color_name,
                                "circularity": round(circularity, 2),
                                "area": int(area),
                            },
                        }
                    )

        return stamps

    # ── 8. Marginal Notes Detection ──
    def detect_marginal_notes(self, gray: np.ndarray) -> list:
        """Detect handwritten notes in page margins.

        Strategy:
            - Check narrow margins (4% of width) for content
            - Use binary threshold to measure actual ink density (not paper texture)
            - Filter by minimum size to avoid noise
            - Confidence based on content density

        Adapted from [1] detect_marginal_notes().
        """
        notes = []
        h, w = gray.shape
        margin_width = int(w * 0.04)  # Narrow margin — 4% of page width

        for region_name, x_start in [("left_margin", 0), ("right_margin", w - margin_width)]:
            margin = gray[:, x_start : x_start + margin_width]

            # Skip nearly empty margins early
            if np.mean(margin) > 250:
                continue

            # Threshold to find actual ink pixels
            _, binary = cv2.threshold(margin, 128, 255, cv2.THRESH_BINARY_INV)

            # Measure content density from binary (ink pixels / total pixels)
            content_density = np.mean(binary > 0) / 255.0

            # Only process margins with meaningful content
            if content_density < 0.05:  # At least 5% ink coverage
                continue

            # Dilate to merge nearby strokes into contiguous regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Find contours of potential note regions
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)

                # Filter by minimum size to avoid noise and table borders
                # ch > 40: taller than a typical line of text
                # cw > 20: wider than a single character
                # ch * cw > 800: minimum area to be considered a real note
                if ch > 40 and cw > 20 and ch * cw > 800:
                    notes.append(
                        {
                            "type": "marginal_note",
                            "bbox": (x_start + cx, cy, x_start + cx + cw, cy + ch),
                            "confidence": round(min(0.8, 0.3 + content_density), 2),
                            "source": "heuristic",
                            "properties": {
                                "region": region_name,
                                "height": ch,
                                "width": cw,
                            },
                        }
                    )

        return notes

    # ── 9. Record Type Assignment ──
    def assign_record_type(self, gray: np.ndarray, blocks: list) -> list:
        """Assign birth/marriage/death type to record blocks.

        Adapted from [1] assign_record_type().
        """
        # Check header region for content density to determine record type
        header = gray[: int(gray.shape[0] * 0.2), :]
        header_density = np.mean(header < 128)

        detected_type = "unknown"
        if header_density > 0.05:
            # If header has significant content, mark as generic
            # In production, OCR would be used here to read actual keywords
            detected_type = "birth"  # Default to birth (most common)

        for block in blocks:
            if block["type"] == "record_block":
                block["properties"]["record_type"] = detected_type

        return blocks

    # ── 10. Reading Order ──
    def sort_reading_order(self, elements: list) -> list:
        """Sort elements in reading order: top-to-bottom, left-to-right.

        Adapted from [1] sort_reading_order().
        """
        return sorted(elements, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    # ── 11. Page Number Detection ──
    def detect_page_number(self, gray: np.ndarray) -> list:
        """Detect page numbers in top or bottom margin."""
        h, w = gray.shape
        page_numbers = []

        # Check top margin
        top_strip = gray[: int(h * 0.08), :]
        top_density = np.mean(top_strip < 128)

        # Check bottom margin
        bottom_strip = gray[int(h * 0.92) :, :]
        bottom_density = np.mean(bottom_strip < 128)

        for region_name, strip, density in [
            ("top_margin", top_strip, top_density),
            ("bottom_margin", bottom_strip, bottom_density),
        ]:
            if density > 0.01 and density < 0.1:
                _, binary = cv2.threshold(strip, 128, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    if 10 < cw < 100 and 10 < ch < 50:
                        y_offset = 0 if region_name == "top_margin" else int(h * 0.92)
                        page_numbers.append(
                            {
                                "type": "page_number",
                                "bbox": (cx, y_offset + cy, cx + cw, y_offset + cy + ch),
                                "confidence": 0.6,
                                "source": "heuristic",
                                "properties": {"region": region_name},
                            }
                        )

        return page_numbers

    # ── 12. Signature Detection ──
    def detect_signatures(self, gray: np.ndarray) -> list:
        """Detect signature regions (dense handwriting at bottom of records)."""
        h, w = gray.shape
        signatures = []

        # Signatures typically appear in the bottom-right of record blocks
        bottom_strip = gray[int(h * 0.85) : int(h * 0.95), int(w * 0.5) :]
        if bottom_strip.size == 0:
            return signatures

        local_var = cv2.Laplacian(bottom_strip, cv2.CV_64F).var()
        edge_density = np.mean(cv2.Canny(bottom_strip, 50, 150) > 0)

        if local_var > 1000 and edge_density > 0.15:
            _, binary = cv2.threshold(bottom_strip, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if cw > 50 and ch > 15 and area > 300:
                    signatures.append(
                        {
                            "type": "signature",
                            "bbox": (
                                int(w * 0.5) + cx,
                                int(h * 0.85) + cy,
                                int(w * 0.5) + cx + cw,
                                int(h * 0.85) + cy + ch,
                            ),
                            "confidence": 0.5,
                            "source": "heuristic",
                            "properties": {},
                        }
                    )

        return signatures

    # ── Main Pipeline ──
    def process(
        self, image: np.ndarray, gray_raw: np.ndarray = None, visualize: bool = False
    ) -> dict:
        """Complete layout analysis pipeline.

        Args:
            image: Light-preprocessed BGR image (from light_preprocess()).
            gray_raw: Optional raw grayscale image (before illumination normalization).
                    If provided, table detection uses this to preserve thin lines.
        """
        gray = _to_gray(image)
        h, w = image.shape[:2]
        all_elements = []

        # 1. Table detection — use raw gray if available (preserves thin table lines)
        if gray_raw is not None:
            table_info = self.detect_table_boundaries_from_raw(gray_raw)
        else:
            table_info = self.detect_table_robust(image, gray_raw)

        # 2. Determine if this is a table page or text page
        is_table = self._is_table_page(gray, table_info)

        if is_table:
            # ── Table page: detect table, cells, record blocks ──
            logger.info(
                f"Detected TABLE page: {table_info['num_rows']}r x {table_info['num_cols']}c"
            )

            # Table boundary
            boundary = {
                "type": "table",
                "bbox": (0, 0, w, h),
                "confidence": self._table_confidence(table_info),
                "source": "projection",
                "properties": {
                    "num_rows": table_info["num_rows"],
                    "num_cols": table_info["num_cols"],
                },
            }
            all_elements.append(boundary)

            # Cells
            cells = self.extract_cells(gray, table_info)
            all_elements.extend(cells)

            # Record blocks
            record_blocks = self.detect_record_blocks(gray, table_info)
            all_elements.extend(record_blocks)

            table_summary = {
                "tables": 1,
                "cells": len(cells),
                "record_blocks": len([e for e in all_elements if e["type"] == "record_block"]),
                "handwritten_cells": sum(
                    1 for c in cells if c["properties"].get("region_type") == "handwritten"
                ),
                "printed_cells": sum(
                    1 for c in cells if c["properties"].get("region_type") == "printed"
                ),
            }
        else:
            # ── Non-table page: detect text blocks ──
            logger.info("Detected TEXT page (no table structure found)")

            text_blocks = self.detect_text_blocks(gray)

            # Add a single text_page marker
            all_elements.append(
                {
                    "type": "text_block",
                    "bbox": (0, 0, w, h),
                    "confidence": 0.8,
                    "source": "page_classification",
                    "properties": {
                        "page_type": "text",
                        "num_blocks": len(text_blocks),
                    },
                }
            )
            all_elements.extend(text_blocks)

            cells = []  # No cells on text pages
            table_summary = {
                "tables": 0,
                "cells": 0,
                "record_blocks": 0,
                "handwritten_cells": 0,
                "printed_cells": 0,
            }

        # 3. Stamps (always check — stamps appear on any page)
        stamps = self.detect_stamps(image)
        all_elements.extend(stamps)

        # 4. Marginal notes (always check)
        notes = self.detect_marginal_notes(gray)
        all_elements.extend(notes)

        # 5. Page numbers (always check)
        page_numbers = self.detect_page_number(gray)
        all_elements.extend(page_numbers)

        # 6. Signatures (always check)
        signatures = self.detect_signatures(gray)
        all_elements.extend(signatures)

        # 7. Sort in reading order
        all_elements = self.sort_reading_order(all_elements)

        # Build result
        result = {
            "elements": all_elements,
            "page_type": "table" if is_table else "text",
            "metadata": {
                "image_size": (w, h),
                "num_elements": len(all_elements),
                "summary": {
                    **table_summary,
                    "stamps": len(stamps),
                    "marginal_notes": len(notes),
                    "page_numbers": len(page_numbers),
                    "signatures": len(signatures),
                    "text_blocks": len([e for e in all_elements if e["type"] == "text_block"]),
                },
            },
        }

        logger.info(
            f"Layout analysis complete: "
            f"{'TABLE' if is_table else 'TEXT'} page, "
            f"{len(all_elements)} elements"
        )

        return result


# ── Simple API ──
def analyze_layout(image: np.ndarray, gray_raw: np.ndarray = None, **kwargs) -> dict:
    """Analyze layout of a light-preprocessed metrical book page.

    Args:
        image: Light-preprocessed BGR image (from light_preprocess()).

    Returns:
        dict with 'elements' list and 'metadata'.
    """
    detector = LayoutDetector()
    return detector.process(image, gray_raw=gray_raw, **kwargs)
