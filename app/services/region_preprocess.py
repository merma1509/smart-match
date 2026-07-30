"""Region-specific preprocessing for document layout components.

Different document regions (text, tables, stamps, signatures) require
different preprocessing strategies for optimal OCR and extraction results.
"""

import cv2
import numpy as np
from loguru import logger


# ── Helpers ──
def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale, return as-is if already gray."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ── Region preprocessors ──
def preprocess_printed_text(region: np.ndarray) -> np.ndarray:
    """Preprocess printed text region for OCR.

    Strategy:
        1. Grayscale conversion
        2. Noise removal (mild denoising)
        3. Adaptive thresholding for binarization
        4. Contrast enhancement with CLAHE

    Best for: Printed text blocks in metrical books.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Mild denoising
    processed = cv2.fastNlMeansDenoising(
        processed, None, h=3, templateWindowSize=7, searchWindowSize=21
    )
    steps.append("denoise(h=3)")

    # 3. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(2.0)")

    # 4. Adaptive threshold
    processed = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    steps.append("adaptive_thresh")

    logger.debug(f"Printed text preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_handwritten(region: np.ndarray) -> np.ndarray:
    """Preprocess handwritten text region for OCR.

    Strategy:
        1. Grayscale conversion
        2. Aggressive denoising (handwriting is noisier)
        3. Contrast enhancement with higher CLAHE
        4. Morphological operations to connect broken strokes
        5. Slight sharpening

    Best for: Handwritten entries in metrical books.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Stronger denoising for handwriting
    processed = cv2.fastNlMeansDenoising(
        processed, None, h=5, templateWindowSize=9, searchWindowSize=21
    )
    steps.append("denoise(h=5)")

    # 3. CLAHE with higher clip limit for handwriting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(3.0)")

    # 4. Morphological operations to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    steps.append("morph_close")

    # 5. Slight sharpening
    blurred = cv2.GaussianBlur(processed, (0, 0), 0.8)
    processed = cv2.addWeighted(processed, 1.3, blurred, -0.3, 0)
    steps.append("sharpen")

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


def preprocess_data_cell(region: np.ndarray) -> np.ndarray:
    """Preprocess a data cell region for OCR.

    Strategy:
        1. Grayscale
        2. Remove cell borders
        3. Contrast enhancement
        4. Mild denoising
        5. Sharpen for better text recognition

    Best for: individual data cells in tables.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Remove border artifacts (3px border)
    h, w = processed.shape
    border = 3
    if h > border * 2 and w > border * 2:
        center = processed[border:-border, border:-border]
        processed = center
        steps.append("border_clean(3px)")

    # 3. Normalize brightness
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    steps.append("norm_brightness")

    # 4. Mild denoising
    processed = cv2.fastNlMeansDenoising(
        processed, None, h=3, templateWindowSize=7, searchWindowSize=21
    )
    steps.append("denoise(h=3)")

    # 5. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(2.0)")

    # 6. Gentle sharpen
    blurred = cv2.GaussianBlur(processed, (0, 0), 1.0)
    processed = cv2.addWeighted(processed, 1.2, blurred, -0.2, 0)
    steps.append("sharpen")
    logger.debug(f"Data cell preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_stamp(region: np.ndarray) -> np.ndarray:
    """Preprocess stamp/seal region.

    Strategy:
        1. Preserve color information (stamps often have meaningful colors)
        2. Denoise while preserving color
        3. Enhance contrast
        4. Sharpen for better detail visibility

    Best for: Official seals and stamps in documents.
    """
    steps = []

    # Preserve color for stamps
    if len(region.shape) == 2:
        processed = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
        steps.append("gray_to_bgr")
    else:
        processed = region.copy()

    # Color denoising
    processed = cv2.fastNlMeansDenoisingColored(
        processed, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )
    steps.append("color_denoise")

    # Contrast enhancement in LAB space
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    light_ch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(light_ch)
    merged = cv2.merge([l_enhanced, a, b])
    processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    steps.append("clahe_color")

    # Sharpen
    blurred = cv2.GaussianBlur(processed, (0, 0), 0.8)
    processed = cv2.addWeighted(processed, 1.3, blurred, -0.3, 0)
    steps.append("sharpen")

    logger.debug(f"Stamp preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_signature(region: np.ndarray) -> np.ndarray:
    """Preprocess signature region.

    Strategy:
        1. Grayscale
        2. Strong contrast enhancement (signatures often faint)
        3. Morphological operations to enhance strokes
        4. Binarization

    Best for: Signature blocks in documents.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. Strong CLAHE for faint signatures
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(4.0)")

    # 3. Morphological operations to enhance strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    steps.append("morph_close")

    # 4. Otsu thresholding
    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps.append("otsu_thresh")

    logger.debug(f"Signature preprocessing: {' -> '.join(steps)}")
    return processed


def preprocess_marginal_note(region: np.ndarray) -> np.ndarray:
    """Preprocess marginal note region.

    Strategy:
        1. Grayscale
        2. Contrast enhancement
        3. Mild denoising
        4. Sharpening

    Best for: Marginal notes and annotations.
    """
    steps = []

    # 1. Grayscale
    processed = _to_grayscale(region)
    steps.append("grayscale")

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    steps.append("clahe(2.5)")

    # 3. Mild denoising
    processed = cv2.fastNlMeansDenoising(processed, None, h=3, templateWindowSize=7, searchWindowSize=21)
    steps.append("denoise(h=3)")

    # 4. Sharpen
    blurred = cv2.GaussianBlur(processed, (0, 0), 1.0)
    processed = cv2.addWeighted(processed, 1.2, blurred, -0.2, 0)
    steps.append("sharpen")

    logger.debug(f"Marginal note preprocessing: {' -> '.join(steps)}")
    return processed


# ── Dispatcher ──
def preprocess_region(region: np.ndarray, region_type: str, **kwargs) -> np.ndarray:
    """Dispatch region to the appropriate preprocessor based on type.

    Args:
        region: Cropped BGR or grayscale image of the region.
        region_type: One of 'printed_text', 'handwritten', 'table_cell',
                     'stamp', 'signature', 'marginal_note', 'data_cell',
                     'header_row', 'text_block', 'record_block'.
        **kwargs: Additional parameters passed to the specific preprocessor.

    Returns:
        Preprocessed image (grayscale for text regions, BGR for stamps).

    Raises:
        ValueError: If region_type is not recognized.
    """
    preprocessors = {
        "printed_text": preprocess_printed_text,
        "handwritten": preprocess_handwritten,
        "table_cell": preprocess_table_cell,
        "stamp": preprocess_stamp,
        "signature": preprocess_signature,
        "marginal_note": preprocess_marginal_note,
        "data_cell": preprocess_data_cell,
        "header_row": preprocess_printed_text,
        "text_block": preprocess_printed_text,
        "record_block": preprocess_table_cell,
    }

    if region_type not in preprocessors:
        logger.warning(f"Unknown region type '{region_type}', using printed_text as fallback")
        region_type = "printed_text"

    preprocessor = preprocessors[region_type]
    return preprocessor(region)
