import numpy as np
from PIL import Image
from skimage import morphology


def load_image_and_mask(image_path, mask_path, config):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {image_path} {image.shape[:2]} "
            f"vs {mask_path} {mask.shape[:2]}"
        )

    threshold = config["svm"]["features"].get("mask_threshold", 127)
    binary_mask = mask > threshold

    if config["svm"]["preprocessing"].get("morphology", False):
        footprint = morphology.square(3)
        binary_mask = morphology.binary_opening(binary_mask, footprint)
        binary_mask = morphology.binary_closing(binary_mask, footprint)

    if not binary_mask.any():
        raise ValueError(f"Empty lesion mask: {mask_path}")

    if config["svm"]["preprocessing"].get("normalize", True):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    return image, binary_mask


def crop_to_mask(image, mask, padding=4):
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, mask.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, mask.shape[1])
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
