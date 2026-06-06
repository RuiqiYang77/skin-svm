"""
Preprocessing v1 + gamma correction — nonlinear brightness adjustment before CLAHE.
Darkens or brightens midtones to help distinguish deep-coloured NV from pale MEL.
Default gamma=1.15 (slight darkening, enhances contrast in dark lesions).
"""
import numpy as np
from PIL import Image
from skimage import color as sk_color, exposure, morphology


def load_image_and_mask(image_path, mask_path, config):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch: {image_path}")
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
        raise ValueError(f"Empty mask: {mask_path}")

    # ★ NEW: Gamma correction (nonlinear brightness)
    if prep_cfg.get("gamma_correction", True):
        gamma = float(prep_cfg.get("gamma_value", 1.15))
        image = exposure.adjust_gamma(image, gamma=gamma)

    if prep_cfg.get("clahe", False):
        lab = sk_color.rgb2lab(image)
        L = lab[:, :, 0] / 100.0
        kernel = prep_cfg.get("clahe_kernel", 12)
        clip = prep_cfg.get("clahe_clip", 0.03)
        L_eq = exposure.equalize_adapthist(L, kernel_size=(kernel, kernel), clip_limit=clip)
        lab[:, :, 0] = L_eq * 100.0
        image = np.clip(sk_color.lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)

    if prep_cfg.get("color_normalize", False):
        image = image.astype(np.float32)
        p = float(prep_cfg.get("shades_of_gray_p", 6))
        illuminant = [np.power(np.mean(np.power(image[:,:,c].astype(np.float64),p)), 1.0/p) for c in range(3)]
        illum_mean = np.mean(illuminant)
        for c in range(3):
            if illuminant[c] > 0:
                image[:,:,c] = np.clip(image[:,:,c]/illuminant[c]*illum_mean, 0, 255)
        image = image.astype(np.uint8)

    if prep_cfg.get("normalize", True):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return image, binary_mask


def crop_to_mask(image, mask, padding=4):
    ys, xs = np.where(mask)
    y0 = max(int(ys.min())-padding, 0); y1 = min(int(ys.max())+padding+1, mask.shape[0])
    x0 = max(int(xs.min())-padding, 0); x1 = min(int(xs.max())+padding+1, mask.shape[1])
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
