"""levels=4 GLCM + skin-only SoG (v14b) — reads gamma & kernel from YAML."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import color
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.dataloader import preprocessing_v14b
from src.dataloader.features import (
    _advanced_color_features, _advanced_shape_features, _advanced_texture_features,
    _clinical_features, _color_features, _hog_features, _melanin_features,
    _shape_features,
)


def _texture_levels(image, mask, feature_cfg):
    """GLCM with configurable levels."""
    gray = color.rgb2gray(image)
    from src.dataloader.preprocessing_v14b import crop_to_mask
    crop_gray, crop_mask = crop_to_mask(gray, mask)

    levels = int(feature_cfg.get("glcm_levels", 4))
    gray_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)
    divisor = max(256 // levels, 1) if levels > 0 else 64
    quantized = (gray_uint8 // divisor).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)

    lesion_values = gray_uint8[crop_mask]
    result = {"gray_mean": float(np.mean(lesion_values)),
              "gray_std": float(np.std(lesion_values))}

    points = int(feature_cfg.get("lbp_points", 24))
    radius = int(feature_cfg.get("lbp_radius", 3))
    lbp = local_binary_pattern(gray_uint8, points, radius, method="uniform")
    lbp_values = lbp[crop_mask]
    lbp_bins = points + 2
    hist, _ = np.histogram(lbp_values, bins=lbp_bins, range=(0, lbp_bins), density=False)
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    for idx, v in enumerate(hist): result[f"lbp_hist_{idx}"] = float(v)

    quantized = np.where(crop_mask, quantized, 0).astype(np.uint8)
    distances = feature_cfg.get("glcm_distances", [1, 2, 4])
    glcm = graycomatrix(quantized, distances=distances,
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        values = graycoprops(glcm, prop)
        result[f"glcm_{prop}_mean"] = float(np.mean(values))
        result[f"glcm_{prop}_std"] = float(np.std(values))
    return result


def extract_features(image_path, mask_path, config):
    image, mask = preprocessing_v14b.load_image_and_mask(image_path, mask_path, config)
    features = {}
    fc = config[config["model"]]["features"]

    if fc.get("use_color", True):
        features.update(_color_features(image, mask, fc))
        if fc.get("use_advanced_color", True):
            features.update(_advanced_color_features(image, mask))
    if fc.get("use_texture", True):
        features.update(_texture_levels(image, mask, fc))
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
    for row in tqdm(metadata_df.itertuples(index=False), total=len(metadata_df), desc="GLCM+v14b"):
        fr = {"image_id": row.image_id, "label": row.label, "image_path": row.image_path,
              "mask_path": row.mask_path, "base_id": row.base_id,
              "is_augmented": row.is_augmented, "augmentation_id": row.augmentation_id}
        fr.update(extract_features(row.image_path, row.mask_path, config))
        rows.append(fr)
    df = pd.DataFrame(rows)
    nc = df.select_dtypes(include=[np.number]).columns
    df[nc] = df[nc].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
