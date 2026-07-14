"""Stage 1 preprocessing — light, structure-preserving

Runs BEFORE layout detection
Goal: improve image quality WITHOUT losing structural information

Does:
  - Orientation correction (deskew)
  - Resize and scaling
  - Color normalization (white balance)
  - Mild denoising (non-destructive)
  - Illumination normalization (remove shadows)

Does NOT:
  - Binarization / thresholding
  - Aggressive morphology
  - Margin cropping (may cut table edges)
  - Resolution blowup (keeps ~2000px max)
"""

import cv2
import numpy as np
from loguru import logger

MAX_IMAGE_DIMENSION = 2000


def _limit_image_size(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Resize and scaling — preserves aspect ratio.
    
    Downsizes oversized scans to prevent YOLO OOM.
    Output ~2000px max dimension.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Resizing {w}x{h} -> {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def _mild_denoise(image: np.ndarray) -> np.ndarray:
    """Non-destructive mild denoising.
    
    Non-Local Means with conservative h=3.
    Removes sensor noise while preserving edges and strokes.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )


def _color_normalize(image: np.ndarray) -> np.ndarray:
    """Color normalization — removes color casts from aged paper.
    
    Uses gray-world assumption: averages of R, G, B channels are equalized.
    Corrects: yellowing, sepia tones, blue scanner casts.
    """
    result = image.copy().astype(np.float32)
    
    mean_b = np.mean(result[:, :, 0])
    mean_g = np.mean(result[:, :, 1])
    mean_r = np.mean(result[:, :, 2])
    overall_mean = (mean_r + mean_g + mean_b) / 3.0
    
    scale_r = overall_mean / max(mean_r, 1.0)
    scale_g = overall_mean / max(mean_g, 1.0)
    scale_b = overall_mean / max(mean_b, 1.0)
    
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
    
    logger.info(f"Color normalized (R:{scale_r:.2f} G:{scale_g:.2f} B:{scale_b:.2f})")
    return result.astype(np.uint8)


def _illumination_normalize(image: np.ndarray) -> np.ndarray:
    """Illumination normalization — removes shadows and uneven lighting.
    
    Estimates per-channel background via large Gaussian blur,
    subtracts it, and re-normalizes.
    Handles: page curvature shadows, vignetting, faded edges.
    """
    result = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape[:2]
    kernel_size = max(31, (min(h, w) // 10) | 1)  # odd, ~10% of smallest dim
    
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        background = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        corrected = cv2.subtract(channel, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        result[:, :, c] = corrected
    
    logger.info(f"Illumination normalized (kernel={kernel_size})")
    return result.astype(np.uint8)


def _mild_clahe(image: np.ndarray) -> np.ndarray:
    """Local contrast enhancement — fine-tunes readability.
    
    Works in LAB space to avoid color distortion.
    clipLimit=1.5 is conservative (default is 2.0).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Orientation correction — straightens skewed pages.
    
    Detects dominant lines via Hough transform and rotates to align.
    Critical for table detection — even 1° skew breaks row/column alignment.
    Only corrects if angle > 0.5° (avoids jitter).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        median_angle = np.median(angles)

        if abs(median_angle) > 10:
            logger.warning(f"Large skew detected: {median_angle:.2f}° - verify image quality")
        if abs(median_angle) < 0.5:
            logger.info(f"Skew {median_angle:.2f}° negligible, skipping")
            return image

        h, w = image.shape[:2]
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
        logger.info(f"Deskewed by {median_angle:.2f}°")
        return deskewed

    logger.info("No skew detected")
    return image


def _remove_borders(image: np.ndarray, border_width: int = 5) -> np.ndarray:
    """Remove thin dark scanner borders from edges.
    
    Conservative — only strips 5px from each edge.
    Prevents YOLO from detecting borders as content.
    """
    result = image.copy()
    h, w = image.shape[:2]
    result[:border_width, :] = 255
    result[h - border_width:, :] = 255
    result[:, :border_width] = 255
    result[:, w - border_width:] = 255
    return result


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
) -> np.ndarray:
    """Light preprocessing pipeline — Stage 1 before layout detection.

    Improves image quality without losing structural information.

    Steps performed in order:
        1. Resize and scaling         <- safe max dimension
        2. Color normalization        <- white balance correction
        3. Illumination normalization <- shadow / gradient removal
        4. Mild denoising             <- non-destructive noise removal
        5. Orientation correction     <- deskew
        6. Border removal             <- scanner edge cleanup

    Args:
        image: Input BGR image
        max_dim: Maximum dimension after resize (default: 2000)
        apply_deskew: Correct page rotation
        apply_resize: Resize oversized images
        apply_color_normalize: White balance correction
        apply_denoise: Mild noise removal
        apply_illumination_normalize: Shadow/gradient removal
        apply_border_removal: Scanner edge cleanup

    Returns:
        BGR image — same color space, same dtype.
        Ready for YOLO layout detection.
    """
    steps: list[str] = []

    # 1. Resize
    if apply_resize:
        image = _limit_image_size(image, max_dim=max_dim)
        steps.append(f"resize({max_dim})")

    # 2. Color normalization (white balance)
    if apply_color_normalize:
        image = _color_normalize(image)
        steps.append("color_normalize")

    # 3. Illumination normalization (shadows)
    if apply_illumination_normalize:
        image = _illumination_normalize(image)
        steps.append("illumination_normalize")

    # 4. Mild CLAHE (local contrast)
    # Note: after illumination norm, subtle CLAHE helps
    if apply_illumination_normalize or apply_color_normalize:
        image = _mild_clahe(image)
        steps.append("mild_clahe")

    # 5. Mild denoising
    if apply_denoise:
        image = _mild_denoise(image)
        steps.append("mild_denoise")

    # 6. Deskew
    if apply_deskew:
        image = _deskew(image)
        steps.append("deskew")

    # 7. Border removal
    if apply_border_removal:
        image = _remove_borders(image)
        steps.append("border_removal")

    logger.info(f"Light preprocessing: {' -> '.join(steps)}")
    return image