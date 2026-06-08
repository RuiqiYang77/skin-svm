"""
Generate new augmentations for domain generalization:
  aug3: Rotation ±15° (random)
  aug4: HSV jitter + Gaussian blur
  aug5: Cutout (random 8-12% occlusion)
"""
import numpy as np
from PIL import Image, ImageEnhance
import os, random

random.seed(42)
np.random.seed(42)

img_dir = "g:/skin/DIP-project/data/image"
mask_dir = "g:/skin/DIP-project/data/mask"
base_ids = list(range(1, 201))  # 200 original images

def load_image(base):
    return np.array(Image.open(f"{img_dir}/{base}.jpg").convert("RGB"))

def load_mask(base):
    return np.array(Image.open(f"{mask_dir}/mask_{base}.jpg").convert("L"))

def save(base, suffix, img_arr, mask_arr):
    Image.fromarray(img_arr).save(f"{img_dir}/{base}_{suffix}.jpg")
    Image.fromarray(mask_arr).save(f"{mask_dir}/mask_{base}_{suffix}.jpg")

def rotate(img, mask):
    from scipy.ndimage import rotate as nd_rotate
    angle = random.uniform(-15, 15)
    r_img = nd_rotate(img, angle, reshape=False, mode="reflect", order=1)
    r_mask = nd_rotate(mask, angle, reshape=False, mode="reflect", order=0)
    return np.clip(r_img, 0, 255).astype(np.uint8), (r_mask > 127).astype(np.uint8) * 255

def hsv_blur(img, mask):
    from skimage import color
    hsv = color.rgb2hsv(img)
    # Hue shift ±5°
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-0.014, 0.014)) % 1.0
    # Saturation scale 0.8-1.2
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 1)
    # Value scale 0.9-1.1
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.9, 1.1), 0, 1)
    img_aug = (color.hsv2rgb(hsv) * 255).astype(np.uint8)
    # Light Gaussian blur
    from scipy.ndimage import gaussian_filter
    sigma = random.uniform(0.3, 0.8)
    for c in range(3):
        img_aug[:, :, c] = gaussian_filter(img_aug[:, :, c], sigma)
    return np.clip(img_aug, 0, 255).astype(np.uint8), mask

def cutout(img, mask):
    h, w = img.shape[:2]
    max_attempts = 10
    for _ in range(max_attempts):
        area_ratio = random.uniform(0.08, 0.12)
        cut_h = int(np.sqrt(area_ratio * h * w * random.uniform(0.8, 1.2)))
        cut_w = int(area_ratio * h * w / cut_h)
        cut_h = min(cut_h, h // 2)
        cut_w = min(cut_w, w // 2)
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)

        mask_before = (mask > 127).sum() if mask.max() > 1 else mask.sum()
        overlap = ((mask[y:y+cut_h, x:x+cut_w] > 127) if mask.max() > 1 else mask[y:y+cut_h, x:x+cut_w]).sum()
        if mask_before - overlap > 50:
            img_aug = img.copy()
            img_aug[y:y+cut_h, x:x+cut_w] = 0
            mask_aug = mask.copy()
            mask_aug[y:y+cut_h, x:x+cut_w] = 0
            return img_aug, mask_aug
    # fallback: no cutout
    return img.copy(), mask.copy()

# ========== Generate ==========
for base in base_ids:
    img = load_image(base)
    msk = load_mask(base)
    msk_bin = (msk > 127).astype(np.uint8) * 255

    for status, (suffix, fn) in enumerate([
        ("aug3", rotate), ("aug4", hsv_blur), ("aug5", cutout)
    ]):
        new_img, new_msk = fn(img, msk_bin)
        save(base, suffix, new_img, new_msk)

    if (base + 1) % 50 == 0:
        print(f"  Done {base+1}/200")

print("All augmentations generated!")
