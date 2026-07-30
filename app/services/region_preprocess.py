# ... existing code ...
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


# ... existing code ...

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
        "table_cell": preprocess_table_cell,
        "stamp": preprocess_stamp,
        "signature": preprocess_signature,
        "marginal_note": preprocess_marginal_note,
        "data_cell": preprocess_data_cell,
        "header_row": preprocess_printed_text,
        "text_block": preprocess_printed_text,
        "record_block": preprocess_table_cell,
    }

# ... existing code ...

