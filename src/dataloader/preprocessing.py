"""Load lesion images and masks with optional image preprocessing.

This module centralizes mask thresholding, morphology, hair removal, contrast
enhancement, color normalization, and lesion cropping helpers.
"""

import numpy as np
from PIL import Image
from skimage import color as sk_color, exposure, morphology


def _remove_hair(image):
    """Remove hair using morphological black-hat on each RGB channel.

    Detects thin dark structures (hair) via black-hat with a linear
    structuring element in three orientations, then inpaints them with
    the local mean of non-hair pixels.
    """
    result = image.copy()
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        hair_mask = np.zeros(channel.shape, dtype=bool)
        for angle_deg in (0, 45, 90):
            se = morphology.disk(2)
            se_len = max(channel.shape) // 30
            se_strip = np.zeros((se_len, se_len), dtype=bool)
            cx = se_len // 2
            cy = se_len // 2
            rad = np.deg2rad(angle_deg)
            dx, dy = np.cos(rad), -np.sin(rad)
            for s in range(-se_len // 2, se_len // 2 + 1):
                y = int(round(cy + s * dy))
                x = int(round(cx + s * dx))
                if 0 <= y < se_len and 0 <= x < se_len:
                    se_strip[y, x] = True
            if se_strip.sum() < 3:
                continue
            blackhat = morphology.black_tophat(channel, se_strip)
            hair_mask |= blackhat > 15
        if hair_mask.any() and not hair_mask.all():
            fill_val = channel[~hair_mask].mean()
            result[:, :, c] = np.where(hair_mask, fill_val, channel)
    return result.astype(np.uint8)


def load_image_and_mask(image_path, mask_path, config):
    """Load an RGB image and binary lesion mask with optional preprocessing."""
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {image_path} {image.shape[:2]} "
            f"vs {mask_path} {mask.shape[:2]}"
        )

    model_key = config["model"]
    feature_cfg = config[model_key]["features"]
    prep_cfg = config[model_key]["preprocessing"]

    threshold = feature_cfg.get("mask_threshold", 127)
    binary_mask = mask > threshold

    if prep_cfg.get("morphology", False):
        footprint = morphology.square(3)
        binary_mask = morphology.binary_opening(binary_mask, footprint)
        binary_mask = morphology.binary_closing(binary_mask, footprint)

    if not binary_mask.any():
        raise ValueError(f"Empty lesion mask: {mask_path}")

    if prep_cfg.get("hair_removal", False):
        image = _remove_hair(image)

    if prep_cfg.get("clahe", False):
        lab = sk_color.rgb2lab(image)
        L = lab[:, :, 0] / 100.0  # normalise L to [0, 1]
        kernel = prep_cfg.get("clahe_kernel", 12)
        clip = prep_cfg.get("clahe_clip", 0.03)
        L_eq = exposure.equalize_adapthist(L, kernel_size=(kernel, kernel), clip_limit=clip)
        lab[:, :, 0] = L_eq * 100.0
        image = np.clip(sk_color.lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)

    if prep_cfg.get("color_normalize", False):
        image = image.astype(np.float32)
        p = float(prep_cfg.get("shades_of_gray_p", 6))
        illuminant = []
        for c in range(3):
            L_c = np.power(np.mean(np.power(image[:, :, c].astype(np.float64), p)), 1.0 / p)
            illuminant.append(L_c)
        illum_mean = np.mean(illuminant)
        for c in range(3):
            if illuminant[c] > 0:
                image[:, :, c] = np.clip(
                    image[:, :, c] / illuminant[c] * illum_mean, 0, 255
                )
        image = image.astype(np.uint8)

    if prep_cfg.get("normalize", True):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    return image, binary_mask


def crop_to_mask(image, mask, padding=4):
    """Crop an image-like array to the lesion bounding box plus padding."""
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, mask.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, mask.shape[1])
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
