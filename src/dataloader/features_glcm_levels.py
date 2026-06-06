"""Configurable GLCM levels + gamma preprocessing (reads gamma_value & clahe_kernel from YAML)."""
import sys
from src.dataloader import preprocessing_gamma
sys.modules["src.dataloader.preprocessing"] = preprocessing_gamma

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import color
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from src.dataloader.features import (
    _advanced_color_features, _advanced_shape_features, _advanced_texture_features,
    _clinical_features, _color_features, _hog_features, _melanin_features,
    _shape_features,
)

def _texture_features_levels(image, mask, feature_cfg):
    """GLCM with configurable levels from YAML."""
    gray = color.rgb2gray(image)
    crop_gray, crop_mask = _crop_to_mask(gray, mask)

    levels = int(feature_cfg.get("glcm_levels", 8))
    gray_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)
    divisor = 256 // levels
    quantized = (gray_uint8 // divisor).astype(np.uint8)
    quantized = np.where(crop_mask, quantized, levels - 1).astype(np.uint8)

    lesion_values = gray_uint8[crop_mask]
    result = {"gray_mean": float(np.mean(lesion_values)),
              "gray_std": float(np.std(lesion_values))}

    # LBP
    points = int(feature_cfg.get("lbp_points", 24))
    radius = int(feature_cfg.get("lbp_radius", 3))
    lbp = local_binary_pattern(gray_uint8, points, radius, method="uniform")
    lbp_values = lbp[crop_mask]
    lbp_bins = points + 2
    hist, _ = np.histogram(lbp_values, bins=lbp_bins, range=(0, lbp_bins), density=False)
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    for idx, v in enumerate(hist): result[f"lbp_hist_{idx}"] = float(v)

    # GLCM
    quantized = np.where(crop_mask, quantized, 0).astype(np.uint8)
    distances = feature_cfg.get("glcm_distances", [1,2,4])
    glcm = graycomatrix(quantized, distances=distances,
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        values = graycoprops(glcm, prop)
        result[f"glcm_{prop}_mean"] = float(np.mean(values))
        result[f"glcm_{prop}_std"] = float(np.std(values))
    return result


# Helper: crop with padding
def _crop_to_mask(image, mask, padding=4):
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return image, mask
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, mask.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, mask.shape[1])
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def extract_features(image_path, mask_path, config):
    image, mask = preprocessing_gamma.load_image_and_mask(image_path, mask_path, config)
    features = {}
    fc = config[config["model"]]["features"]

    if fc.get("use_color", True):
        features.update(_color_features(image, mask, fc))
        if fc.get("use_advanced_color", True):
            features.update(_advanced_color_features(image, mask))
    if fc.get("use_texture", True):
        features.update(_texture_features_levels(image, mask, fc))
        if fc.get("use_advanced_texture", True):
            features.update(_advanced_texture_features(image, mask, fc))
    if fc.get("use_shape", True):
        features.update(_shape_features(mask))
        if fc.get("use_advanced_shape", True):
            features.update(_advanced_shape_features(image, mask))
    if fc.get("use_clinical", False):
        features.update(_clinical_features(image, mask))
    if fc.get("use_melanin_features", False):
        features.update(_melanin_features(image, mask))
    if fc.get("use_hog", False):
        features.update(_hog_features(image, mask))
    return features


def extract_feature_table(metadata_df, config):
    rows = []
    for row in tqdm(metadata_df.itertuples(index=False), total=len(metadata_df), desc=f"GLCM levels"):
        fr = {"image_id": row.image_id, "label": row.label, "image_path": row.image_path,
              "mask_path": row.mask_path, "base_id": row.base_id,
              "is_augmented": row.is_augmented, "augmentation_id": row.augmentation_id}
        fr.update(extract_features(row.image_path, row.mask_path, config))
        rows.append(fr)
    df = pd.DataFrame(rows)
    nc = df.select_dtypes(include=[np.number]).columns
    df[nc] = df[nc].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
