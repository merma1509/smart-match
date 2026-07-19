"""Stage 1 preprocessing — light, structure-preserving
Runs BEFORE layout detection
Goal: improve image quality WITHOUT losing structural information

Does:
  - Orientation correction (deskew) — via long-line filtering
  - Resize and scaling — adaptive to content
  - Color normalization (white balance) — gray-world
  - Mild denoising (non-destructive) — NLM
  - Illumination normalization (remove shadows) — background subtraction
  - Adaptive contrast enhancement — CLAHE in LAB
  - Optional Sauvola binarization — for very poor contrast

Does NOT:
  - Aggressive morphology
  - Margin cropping (may cut table edges)
"""
import cv2
import numpy as np
from loguru import logger

# ── Constants ──
MAX_IMAGE_DIMENSION = 3000  # Increased from 2000 for better OCR on small text
MIN_IMAGE_DIMENSION = 1500  # Upscale very small images


def _limit_image_size(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Adaptive resize — preserves aspect ratio.
    - Downsizes oversized scans to prevent OOM
    - Upscales very small scans for better OCR
    Output ~3000px max dimension.
    """
    h, w = image.shape[:2]
    original_shape = (w, h)
    
    # Upscale very small images
    if max(h, w) < MIN_IMAGE_DIMENSION:
        scale = MIN_IMAGE_DIMENSION / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Upscaling {w}x{h} -> {new_w}x{new_h} (was too small)")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Downscale oversized images
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Resizing {w}x{h} -> {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def _deskew_robust(image: np.ndarray) -> np.ndarray:
    """Robust orientation correction — filters long lines only.
    
    Improved over simple Hough: only considers lines that span
    at least 50% of the image width/height. This avoids false
    angles from text lines or noise.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(min(w, h) * 0.15))
    
    if lines is None:
        logger.info("No skew detected (no lines found)")
        return image
    
    # Filter: only long lines (at least 50% of image dimension)
    valid_angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        
        # Calculate line length in image
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Find intersection points with image borders
        pts = []
        for x in [0, w]:
            y = int((rho - x * a) / (b + 1e-10))
            if 0 <= y <= h:
                pts.append((x, y))
        for y in [0, h]:
            x = int((rho - y * b) / (a + 1e-10))
            if 0 <= x <= w:
                pts.append((x, y))
        
        if len(pts) >= 2:
            length = np.sqrt((pts[0][0] - pts[1][0])**2 + (pts[0][1] - pts[1][1])**2)
            min_length = max(w, h) * 0.5  # At least 50% of image
            if length >= min_length:
                valid_angles.append(angle)
    
    if not valid_angles:
        logger.info("No long lines found, skipping deskew")
        return image
    
    median_angle = np.median(valid_angles)
    
    if abs(median_angle) > 10:
        logger.warning(f"Large skew detected: {median_angle:.2f}° — verify image quality")
    
    if abs(median_angle) < 0.3:
        logger.info(f"Skew {median_angle:.2f}° negligible, skipping")
        return image
    
    # Rotate
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    deskewed = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.info(f"Deskewed by {median_angle:.2f}° (from {len(valid_angles)} long lines)")
    return deskewed


def _color_normalize(image: np.ndarray) -> np.ndarray:
    """Color normalization — removes color casts from aged paper.
    Uses gray-world assumption with clipping to avoid over-correction.
    """
    result = image.copy().astype(np.float32)
    
    mean_b = np.mean(result[:, :, 0])
    mean_g = np.mean(result[:, :, 1])
    mean_r = np.mean(result[:, :, 2])
    
    overall_mean = (mean_r + mean_g + mean_b) / 3.0
    
    # Clip scale factors to [0.7, 1.3] to avoid over-correction
    scale_r = np.clip(overall_mean / max(mean_r, 1.0), 0.7, 1.3)
    scale_g = np.clip(overall_mean / max(mean_g, 1.0), 0.7, 1.3)
    scale_b = np.clip(overall_mean / max(mean_b, 1.0), 0.7, 1.3)
    
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
    
    logger.info(f"Color normalized (R:{scale_r:.2f} G:{scale_g:.2f} B:{scale_b:.2f})")
    return result.astype(np.uint8)


def _illumination_normalize(image: np.ndarray) -> np.ndarray:
    """Illumination normalization — removes shadows and uneven lighting.
    Uses large Gaussian blur for background estimation.
    """
    result = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape[:2]
    
    # Adaptive kernel size: larger for bigger images
    kernel_size = max(61, (min(h, w) // 8) | 1)
    
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        background = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        corrected = cv2.subtract(channel, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        result[:, :, c] = corrected
    
    logger.info(f"Illumination normalized (kernel={kernel_size})")
    return result.astype(np.uint8)


def _enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Adaptive contrast enhancement — CLAHE in LAB space.
    clip_limit: higher = more contrast (default 2.0, range 1.0-4.0)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _mild_denoise(image: np.ndarray, h: int = 3) -> np.ndarray:
    """Non-destructive mild denoising.
    h=3 for mild, h=5 for stronger.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=h, hColor=h, 
        templateWindowSize=7, searchWindowSize=21
    )


def _remove_borders(image: np.ndarray, border_width: int = 10) -> np.ndarray:
    """Remove scanner borders from edges.
    Increased from 5 to 10px for better coverage.
    """
    result = image.copy()
    h, w = image.shape[:2]
    result[:border_width, :] = 255
    result[h - border_width:, :] = 255
    result[:, :border_width] = 255
    result[:, w - border_width:] = 255
    return result


def _compute_quality_metrics(image: np.ndarray) -> dict:
    """Compute image quality metrics for diagnostics."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Contrast (std of pixel values)
    contrast = float(np.std(gray))
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))
    
    # Entropy (texture complexity)
    entropy = float(-np.sum(
        (gray / 255.0 + 1e-10) * np.log(gray / 255.0 + 1e-10)
    ))
    
    return {
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "entropy": round(entropy, 2),
    }


# ── Public API ──
def light_preprocess(
    image: np.ndarray,
    *,
    max_dim: int = MAX_IMAGE_DIMENSION,
    apply_deskew: bool = True,
    apply_resize: bool = True,
    apply_color_normalize: bool = True,
    apply_denoise: bool = True,
    apply_illumination_normalize: bool = True,
    apply_border_removal: bool = True,
    apply_contrast_enhance: bool = True,
    contrast_clip_limit: float = 2.0,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Light preprocessing pipeline — Stage 1 before layout detection.
    
    Improved pipeline with better deskew, adaptive resizing,
    and optional quality metrics.
    
    Steps performed in order:
        1. Resize and scaling         <- adaptive (upscale small, downscale large)
        2. Color normalization        <- white balance correction (clipped)
        3. Illumination normalization <- shadow / gradient removal (larger kernel)
        4. Mild denoising             <- non-destructive (NLM)
        5. Orientation correction     <- robust deskew (long lines only)
        6. Contrast enhancement       <- CLAHE in LAB
        7. Border removal             <- scanner edge cleanup (10px)
    
    Args:
        image: Input BGR image (numpy array)
        max_dim: Maximum dimension after resize (default: 3000)
        apply_deskew: Correct page rotation
        apply_resize: Resize oversized images / upscale small ones
        apply_color_normalize: White balance correction
        apply_denoise: Mild noise removal
        apply_illumination_normalize: Shadow/gradient removal
        apply_border_removal: Scanner edge cleanup
        apply_contrast_enhance: CLAHE contrast enhancement
        contrast_clip_limit: CLAHE clip limit (1.0-4.0, default 2.0)
        return_metrics: If True, return (image, metrics_dict)
    
    Returns:
        BGR image or (BGR image, metrics dict)
    """
    steps: list[str] = []
    before_metrics = _compute_quality_metrics(image) if return_metrics else None
    
    # 1. Resize (adaptive)
    if apply_resize:
        image = _limit_image_size(image, max_dim=max_dim)
        steps.append(f"resize({max_dim})")
    
    # 2. Color normalization
    if apply_color_normalize:
        image = _color_normalize(image)
        steps.append("color_normalize")
    
    # 3. Illumination normalization
    if apply_illumination_normalize:
        image = _illumination_normalize(image)
        steps.append("illumination_normalize")
    
    # 4. Mild denoising (before deskew to reduce noise in Hough)
    if apply_denoise:
        image = _mild_denoise(image, h=3)
        steps.append("mild_denoise")
    
    # 5. Robust deskew (after denoise for cleaner edge detection)
    if apply_deskew:
        image = _deskew_robust(image)
        steps.append("deskew_robust")
    
    # 6. Contrast enhancement (after deskew for cleaner results)
    if apply_contrast_enhance:
        image = _enhance_contrast(image, clip_limit=contrast_clip_limit)
        steps.append(f"contrast(clip={contrast_clip_limit})")
    
    # 7. Border removal
    if apply_border_removal:
        image = _remove_borders(image, border_width=10)
        steps.append("border_removal")
    
    after_metrics = _compute_quality_metrics(image) if return_metrics else None
    
    if return_metrics:
        logger.info(f"Preprocessing: {' -> '.join(steps)}")
        logger.info(f"Quality: contrast {before_metrics['contrast']}->{after_metrics['contrast']}, "
                   f"sharpness {before_metrics['sharpness']}->{after_metrics['sharpness']}")
        return image, {"before": before_metrics, "after": after_metrics, "steps": steps}
    
    logger.info(f"Light preprocessing: {' -> '.join(steps)}")
    return image