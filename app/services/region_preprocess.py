"""Region-Specific Preprocessing — Stage 2b after layout detection.

Takes individual cropped regions from layout detection and applies
optimized preprocessing based on region type:

    - printed_text:     Contrast enhancement + adaptive thresholding + sharpening
    - handwritten:      Denoising + stroke enhancement + background smoothing
    - table_cell:       Line enhancement + cell boundary cleanup
    - stamp:            Color preservation + circular enhancement
    - signature:        Stroke enhancement + high-pass filtering
    - marginal_note:    Contrast stretch + adaptive threshold

Each function returns a preprocessed grayscale image ready for OCR.
"""


import cv2
import numpy as np
from loguru import logger


# ── Shared Helpers
def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _limit_image_size(image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Downsize if too large."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


# ── Region-Type Specific Preprocessors ──
def preprocess_printed_text(region: np.ndarray) -> np.ndarray:
    """Preprocess a printed text region for TrOCR.

    KEY: TrOCR models work with grayscale, NOT binarized images!
    Adaptive thresholding DESTROYS text quality for TrOCR.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Gentle CLAHE (lower clip limit)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(1.5)")

    # 3. Denoise (mild, preserve edges)
    processed = cv2.fastNlMeansDenoising(
        processed, None, h=5, templateWindowSize=7, searchWindowSize=21
    )
    steps.append("denoise(h=5)")

    # 4. Gentle sharpen
    blurred = cv2.GaussianBlur(processed, (0, 0), 1.0)
    sharpened = cv2.addWeighted(processed, 1.3, blurred, -0.3, 0)
    processed = sharpened
    steps.append("sharpen")

    # 5. NO binarization! TrOCR needs grayscale
    # NOT: cv2.adaptiveThreshold(...)

    logger.debug(f"Printed text preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_handwritten(region: np.ndarray) -> np.ndarray:
    """Preprocess handwritten region for TrOCR.

    TrOCR models pre-trained on handwritten text expect
    the image to look like the training data — grayscale,
    slightly denoised, but NOT binarized.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Gentle denoise
    processed = cv2.fastNlMeansDenoising(
        processed, None, h=3, templateWindowSize=7, searchWindowSize=21
    )
    steps.append("denoise(h=3)")

    # 3. Gentle CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(1.5)")

    # 4. NO aggressive morphology, NO background subtraction
    # TrOCR handles these internally

    logger.debug(f"Handwritten preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_table_cell(region: np.ndarray) -> np.ndarray:
    """Preprocess a table cell region.

    Strategy:
        1. Grayscale
        2. Detect cell boundaries
        3. Clean border artifacts
        4. Contrast enhancement
        5. Optional binarization for printed cells

    Best for: individual table cells after grid detection.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Remove border artifacts (dark frame around cell)
    h, w = processed.shape
    border = 2
    if h > border * 2 and w > border * 2:
        center = processed[border:-border, border:-border]
        processed = cv2.copyMakeBorder(center, border, border, border, border, cv2.BORDER_REPLICATE)
    steps.append("border_clean")

    # 3. Normalize brightness
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    steps.append("norm_brightness")

    # 4. Mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(2.0)")

    logger.debug(f"Table cell preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_stamp(region: np.ndarray) -> np.ndarray:
    """Preprocess a stamp/seal region.

    Strategy:
        1. Keep color information (BGR output)
        2. Enhance saturation for stamp color
        3. Circular edge enhancement
        4. Return both color and grayscale versions

    Best for: red/blue stamps, official seals.
    Returns: BGR image (color preserved).
    """
    steps = []

    # 1. Ensure BGR
    if len(region.shape) == 2:
        processed = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    else:
        processed = region.copy()
    steps.append("bgr")

    # 2. Enhance saturation
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 30)  # Increase saturation
    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    steps.append("saturation_boost")

    # 3. Sharpen edges
    blurred = cv2.GaussianBlur(processed, (0, 0), 2.0)
    processed = cv2.addWeighted(processed, 1.3, blurred, -0.3, 0)
    steps.append("sharpen")

    logger.debug(f"Stamp preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_signature(region: np.ndarray) -> np.ndarray:
    """Preprocess a signature region for analysis.

    Strategy:
        1. Grayscale
        2. High-pass filtering to emphasize stroke changes
        3. Stroke thickening
        4. Contrast maximization

    Best for: signature verification, stroke analysis.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. High-pass filter
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    highpass = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)
    processed = highpass
    steps.append("highpass")

    # 3. Stroke thickening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    processed = binary
    steps.append("stroke_thicken")

    logger.debug(f"Signature preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_marginal_note(region: np.ndarray) -> np.ndarray:
    """Preprocess a marginal note region.

    Strategy:
        1. Grayscale
        2. Aggressive contrast stretch (margins are often faded)
        3. Adaptive thresholding
        4. Noise removal

    Best for: handwritten notes in page margins.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Aggressive contrast stretch
    p2, p98 = np.percentile(processed, (2, 98))
    processed = np.clip((processed - p2) * (255.0 / (p98 - p2 + 1e-6)), 0, 255).astype(np.uint8)
    steps.append("contrast_stretch")

    # 3. Adaptive threshold
    processed = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    steps.append("adaptive_threshold(15,3)")

    # 4. Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    steps.append("noise_removal")

    logger.debug(f"Marginal note preprocessing: {' -> '.join(steps)}")
    return processed


# ── Dispatcher ──
def preprocess_region(region: np.ndarray, region_type: str, **kwargs) -> np.ndarray:
    """Dispatch region to the appropriate preprocessor based on type.

    Args:
        region: Cropped BGR or grayscale image of the region.
        region_type: One of 'printed_text', 'handwritten', 'table_cell',
                     'stamp', 'signature', 'marginal_note'.
        **kwargs: Additional parameters passed to the specific preprocessor.

    Returns:
        Preprocessed image (grayscale for text regions, BGR for stamps).
    """
    preprocessors = {
        "printed_text": preprocess_printed_text,
        "handwritten": preprocess_handwritten,
        "table_cell": None,
        "stamp": preprocess_stamp,
        "signature": preprocess_signature,
        "marginal_note": preprocess_marginal_note,
        "data_cell": None,
        "header_row": preprocess_printed_text,
        "text_block": preprocess_printed_text,
        "record_block": preprocess_table_cell,
    }

    preprocessor = preprocessors.get(region_type)
    if preprocessor is None:
        logger.warning(f"Unknown region type '{region_type}', using default grayscale conversion")
        return _to_grayscale(region)

    result = preprocessor(region, **kwargs)
    logger.info(f"Region preprocessing applied: {region_type}")
    return result


def preprocess_regions(regions: list, full_image: np.ndarray) -> list:
    """Preprocess multiple regions from layout detection.

    Args:
        regions: List of region dicts from layout detection, each with
                 'type' and 'bbox' keys.
        full_image: Original (or light-preprocessed) full-page BGR image.

    Returns:
        List of dicts with original metadata plus 'preprocessed' image.
    """
    results = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        cropped = full_image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Use region_type property for data_cells to pick handwritten vs printed
        region_type = region["type"]
        if region_type == "data_cell":
            props = region.get("properties", {})
            cell_type = props.get("region_type", "printed")
            if cell_type == "handwritten":
                region_type = "handwritten"
            else:
                region_type = "printed_text"

        preprocessed = preprocess_region(cropped, region_type)
        results.append({**region, "original_region": cropped, "preprocessed": preprocessed})

    return results


# ── Simple API ──
def process_region(region: np.ndarray, region_type: str) -> np.ndarray:
    """Simple API: preprocess a single region.

    Args:
        region: Cropped BGR or grayscale image.
        region_type: 'printed_text', 'handwritten', 'table_cell',
                     'stamp', 'signature', or 'marginal_note'.

    Returns:
        Preprocessed image ready for OCR.
    """
    return preprocess_region(region, region_type)
