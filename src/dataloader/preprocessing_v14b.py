"""
V14b — skin-only SoG: estimate illuminant from healthy skin pixels ONLY.
Excludes lesion (mask=True) and black borders from the Minkowski norm.
This correctly decouples light source estimation from diagnostic lesion colour.
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
    fc = config[model_key]["features"]; pc = config[model_key]["preprocessing"]
    binary_mask = mask > fc.get("mask_threshold", 127)
    if pc.get("morphology", False):
        fp = morphology.square(3)
        binary_mask = morphology.binary_opening(binary_mask, fp)
        binary_mask = morphology.binary_closing(binary_mask, fp)
    if not binary_mask.any(): raise ValueError(f"Empty mask: {mask_path}")

    if pc.get("gamma_correction", True):
        image = exposure.adjust_gamma(image, gamma=float(pc.get("gamma_value", 1.08)))

    if pc.get("clahe", False):
        lab = sk_color.rgb2lab(image); L = lab[:,:,0]/100.0
        kernel = pc.get("clahe_kernel", 8)
        L_eq = exposure.equalize_adapthist(L, kernel_size=(kernel,)*2, clip_limit=pc.get("clahe_clip",0.03))
        lab[:,:,0] = L_eq*100.0; image = np.clip(sk_color.lab2rgb(lab)*255,0,255).astype(np.uint8)

    # ★ SKIN-ONLY SoG: exclude lesion + black borders from illuminant estimation
    if pc.get("color_normalize", False):
        image_f = image.astype(np.float32)
        # Define "healthy skin": NOT lesion AND NOT black border
        is_black_border = (image[:,:,0] < 15) & (image[:,:,1] < 15) & (image[:,:,2] < 15)
        skin_valid = ~binary_mask & ~is_black_border

        if skin_valid.sum() < 100:
            # Fallback: use all non-lesion pixels (original SoG minus lesion)
            skin_valid = ~binary_mask

        p = float(pc.get("shades_of_gray_p", 6))
        illum = []
        for c in range(3):
            vals = image_f[:,:,c][skin_valid].astype(np.float64)
            if len(vals) > 0:
                L_c = np.power(np.mean(np.power(vals, p)), 1.0/p)
            else:
                L_c = np.power(np.mean(np.power(image_f[:,:,c].astype(np.float64), p)), 1.0/p)
            illum.append(L_c)
        im = np.mean(illum)
        for c in range(3):
            if illum[c] > 0: image_f[:,:,c] = np.clip(image_f[:,:,c]/illum[c]*im, 0, 255)
        image = image_f.astype(np.uint8)

    image = image.astype(np.float32)/255.0 if pc.get("normalize", True) else image.astype(np.float32)
    return image, binary_mask

def crop_to_mask(image, mask, padding=4):
    ys,xs=np.where(mask); y0=max(int(ys.min())-padding,0);y1=min(int(ys.max())+padding+1,mask.shape[0]);x0=max(int(xs.min())-padding,0);x1=min(int(xs.max())+padding+1,mask.shape[1])
    return image[y0:y1,x0:x1], mask[y0:y1,x0:x1]
