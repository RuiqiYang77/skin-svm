import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "dip_project_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage import color, measure
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.transform import resize
from tqdm import tqdm

from src.dataloader.preprocessing import crop_to_mask, load_image_and_mask


def extract_features(image_path, mask_path, config):
    image, mask = load_image_and_mask(image_path, mask_path, config)
    features = {}
    feature_cfg = config["svm"]["features"]

    if feature_cfg.get("use_color", True):
        features.update(_color_features(image, mask, feature_cfg))

    if feature_cfg.get("use_texture", True):
        features.update(_texture_features(image, mask, feature_cfg))

    if feature_cfg.get("use_shape", True):
        features.update(_shape_features(mask))

    if feature_cfg.get("use_hog", False):
        features.update(_hog_features(image, mask))

    return features


def extract_feature_table(metadata_df, config):
    rows = []
    for row in tqdm(
        metadata_df.itertuples(index=False),
        total=len(metadata_df),
        desc="Extracting features",
    ):
        feature_row = {
            "image_id": row.image_id,
            "label": row.label,
            "image_path": row.image_path,
            "mask_path": row.mask_path,
            "base_id": row.base_id,
            "is_augmented": row.is_augmented,
            "augmentation_id": row.augmentation_id,
        }
        feature_row.update(extract_features(row.image_path, row.mask_path, config))
        rows.append(feature_row)

    features_df = pd.DataFrame(rows)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0.0)
    return features_df


def _channel_moments(values, prefix):
    result = {}
    for idx in range(values.shape[1]):
        channel = values[:, idx]
        result[f"{prefix}_{idx}_mean"] = float(np.mean(channel))
        result[f"{prefix}_{idx}_std"] = float(np.std(channel))
        result[f"{prefix}_{idx}_skew"] = float(skew(channel, bias=False)) if len(channel) > 2 else 0.0
    return result


def _color_features(image, mask, feature_cfg):
    lesion_rgb = image[mask]
    hsv = color.rgb2hsv(image)
    lesion_hsv = hsv[mask]

    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]

    result = {}
    result.update(_channel_moments(lesion_rgb, "rgb"))
    result.update(_channel_moments(lesion_hsv, "hsv"))
    result.update(_channel_moments(lesion_lab, "lab"))

    bins = int(feature_cfg.get("hsv_hist_bins", 16))
    for channel_idx, channel_name in enumerate(["h", "s", "v"]):
        hist, _ = np.histogram(
            lesion_hsv[:, channel_idx], bins=bins, range=(0.0, 1.0), density=False
        )
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        for bin_idx, value in enumerate(hist):
            result[f"hsv_{channel_name}_hist_{bin_idx}"] = float(value)

    result["dark_pixel_ratio"] = float(np.mean(lesion_hsv[:, 2] < 0.35))
    result["hue_std"] = float(np.std(lesion_hsv[:, 0]))
    result["saturation_mean"] = float(np.mean(lesion_hsv[:, 1]))
    return result


def _texture_features(image, mask, feature_cfg):
    gray = color.rgb2gray(image)
    crop_gray, crop_mask = crop_to_mask(gray, mask)

    gray_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)
    lesion_values = gray_uint8[crop_mask]
    result = {
        "gray_mean": float(np.mean(lesion_values)),
        "gray_std": float(np.std(lesion_values)),
    }

    points = int(feature_cfg.get("lbp_points", 24))
    radius = int(feature_cfg.get("lbp_radius", 3))
    lbp = local_binary_pattern(gray_uint8, points, radius, method="uniform")
    lbp_values = lbp[crop_mask]
    lbp_bins = points + 2
    hist, _ = np.histogram(lbp_values, bins=lbp_bins, range=(0, lbp_bins), density=False)
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    for idx, value in enumerate(hist):
        result[f"lbp_hist_{idx}"] = float(value)

    quantized = (gray_uint8 // 32).astype(np.uint8)
    quantized = np.where(crop_mask, quantized, 0).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        values = graycoprops(glcm, prop)
        result[f"glcm_{prop}_mean"] = float(np.mean(values))
        result[f"glcm_{prop}_std"] = float(np.std(values))

    return result


def _shape_features(mask):
    label_img = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(label_img)
    if not regions:
        return {
            "area_ratio": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "eccentricity": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "bbox_aspect_ratio": 0.0,
            "border_irregularity": 0.0,
            "horizontal_asymmetry": 0.0,
            "vertical_asymmetry": 0.0,
        }

    region = max(regions, key=lambda r: r.area)
    area = float(region.area)
    perimeter = float(measure.perimeter(mask))
    circularity = 4.0 * np.pi * area / (perimeter**2 + 1e-8)
    minr, minc, maxr, maxc = region.bbox
    height = max(maxr - minr, 1)
    width = max(maxc - minc, 1)

    convex = measure.regionprops(measure.label(region.convex_image.astype(np.uint8)))[0]
    convex_perimeter = float(measure.perimeter(region.convex_image))

    return {
        "area_ratio": area / float(mask.size),
        "perimeter": perimeter,
        "circularity": float(circularity),
        "eccentricity": float(region.eccentricity),
        "major_axis_length": float(region.major_axis_length),
        "minor_axis_length": float(region.minor_axis_length),
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "bbox_aspect_ratio": float(width / height),
        "border_irregularity": float(perimeter / (convex_perimeter + 1e-8)),
        "compactness": float(perimeter / (np.sqrt(area) + 1e-8)),
        "convex_area_ratio": float(area / (convex.area + 1e-8)),
        "horizontal_asymmetry": _asymmetry(mask, axis=1),
        "vertical_asymmetry": _asymmetry(mask, axis=0),
    }


def _asymmetry(mask, axis):
    flipped = np.flip(mask, axis=axis)
    union = np.logical_or(mask, flipped).sum()
    if union == 0:
        return 0.0
    diff = np.logical_xor(mask, flipped).sum()
    return float(diff / union)


def _hog_features(image, mask):
    crop_img, crop_mask = crop_to_mask(image, mask)
    crop_img = resize(
        crop_img,
        (64, 64, 3),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    crop_mask = resize(
        crop_mask.astype(float),
        (64, 64),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    gray = color.rgb2gray(crop_img)
    gray = np.where(crop_mask, gray, 0.0)
    values = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return {f"hog_{idx}": float(value) for idx, value in enumerate(values)}
