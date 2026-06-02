"""Extract handcrafted lesion features for model training and inference.

The module converts an image and lesion mask into tabular color, texture,
shape, and optional dermoscopic descriptors used by the classical classifiers.
"""

import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "dip_project_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter1d, uniform_filter
from scipy.spatial import KDTree
from scipy.stats import skew
from skimage import color, filters, measure, morphology
from skimage.feature import blob_log, graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
from skimage.transform import resize
from tqdm import tqdm

from src.dataloader.preprocessing import crop_to_mask, load_image_and_mask


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features(image_path, mask_path, config):
    """Extract the configured handcrafted feature vector for one sample."""
    image, mask = load_image_and_mask(image_path, mask_path, config)
    features = {}
    feature_cfg = config[config["model"]]["features"]

    if feature_cfg.get("use_color", True):
        features.update(_color_features(image, mask, feature_cfg))
        if feature_cfg.get("use_advanced_color", True):
            features.update(_advanced_color_features(image, mask))

    if feature_cfg.get("use_texture", True):
        features.update(_texture_features(image, mask, feature_cfg))
        if feature_cfg.get("use_advanced_texture", True):
            features.update(_advanced_texture_features(image, mask, feature_cfg))

    if feature_cfg.get("use_shape", True):
        features.update(_shape_features(mask))
        if feature_cfg.get("use_advanced_shape", True):
            features.update(_advanced_shape_features(image, mask))

    if feature_cfg.get("use_clinical", False):
        features.update(_clinical_features(image, mask))

    if feature_cfg.get("use_melanin_features", False):
        features.update(_melanin_features(image, mask))

    if feature_cfg.get("use_hog", False):
        features.update(_hog_features(image, mask))

    if feature_cfg.get("use_dermoscopic_color", False):
        features.update(_dermoscopic_color_features(image, mask))

    if feature_cfg.get("use_dermoscopic_asymmetry", False):
        features.update(_dermoscopic_asymmetry_features(image, mask))

    if feature_cfg.get("use_dermoscopic_border", False):
        features.update(_dermoscopic_border_features(image, mask))

    if feature_cfg.get("use_pigment_network", False):
        features.update(_pigment_network_features(image, mask))

    if feature_cfg.get("use_dots_globules", False):
        features.update(_dots_globules_features(image, mask))

    if feature_cfg.get("use_streaks", False):
        features.update(_streak_features(image, mask))

    if feature_cfg.get("use_regression_structures", False):
        features.update(_regression_structures_features(image, mask))

    if feature_cfg.get("use_composite_features", False):
        features.update(_composite_features(image, mask))

    return features


def extract_feature_table(metadata_df, config):
    """Extract features for every metadata row and return a clean DataFrame."""
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


# ---------------------------------------------------------------------------
# Colour features (original + advanced)
# ---------------------------------------------------------------------------

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


def _advanced_color_features(image, mask):
    """Targeted features for NV vs MEL discrimination."""
    hsv = color.rgb2hsv(image)
    lesion_hsv = hsv[mask]
    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]
    lesion_rgb = image[mask]

    result = {}

    # ---- 1. Blue-white veil ratio (highly specific to MEL) ----------------
    # Blue-white veil: high L, low |a|, low |b|  →  hazy blue-grey-white
    bw_mask = (
        (lesion_lab[:, 0] > 70) &
        (lesion_lab[:, 1] > -15) & (lesion_lab[:, 1] < 15) &
        (lesion_lab[:, 2] > -20) & (lesion_lab[:, 2] < 20)
    )
    result["blue_white_veil_ratio"] = float(bw_mask.mean())

    # ---- 2. Colour variegation (local StdDev pooled over lesion) ---------
    for idx, name in enumerate(["r", "g", "b"]):
        channel = image[:, :, idx]
        local_std = _local_std(channel, mask, window_size=9)
        result[f"color_variegation_{name}"] = float(local_std)

    # ---- 3. Colour entropy (quantised HSV distribution breadth) ----------
    for idx, name in enumerate(["h", "s", "v"]):
        hist, _ = np.histogram(lesion_hsv[:, idx], bins=16, range=(0, 1), density=True)
        hist = hist + 1e-10
        ent = -np.sum(hist * np.log2(hist))
        result[f"color_entropy_{name}"] = float(ent)

    # ---- 4. Dark-spot ratio (very dark globules) -------------------------
    result["dark_spot_ratio"] = float(np.mean(lesion_hsv[:, 2] < 0.2))

    # ---- 5. Centre / periphery colour difference -------------------------
    center_mask = _center_region(mask, shrink_ratio=0.5)
    peripheral_mask = mask & ~center_mask
    if peripheral_mask.sum() > 50 and center_mask.sum() > 50:
        for idx, name in enumerate(["r", "g", "b"]):
            c_mean = image[:, :, idx][center_mask].mean()
            p_mean = image[:, :, idx][peripheral_mask].mean()
            result[f"center_periphery_diff_{name}"] = float(abs(c_mean - p_mean))
    else:
        for idx, name in enumerate(["r", "g", "b"]):
            result[f"center_periphery_diff_{name}"] = 0.0

    # ---- 6. 90th-percentile intensity (captures bright regions) ----------
    for idx, name in enumerate(["r", "g", "b"]):
        result[f"rgb_{name}_p90"] = float(np.percentile(lesion_rgb[:, idx], 90))
        result[f"rgb_{name}_p10"] = float(np.percentile(lesion_rgb[:, idx], 10))
        result[f"rgb_{name}_range"] = float(
            np.percentile(lesion_rgb[:, idx], 95) - np.percentile(lesion_rgb[:, idx], 5)
        )

    # ---- 7. Colour ratio features (inter-channel relationships) ----------
    r_ch, g_ch, b_ch = lesion_rgb[:, 0], lesion_rgb[:, 1], lesion_rgb[:, 2]
    result["rgb_rg_ratio_mean"] = float(np.mean(r_ch / (g_ch + 1e-8)))
    result["rgb_rb_ratio_mean"] = float(np.mean(r_ch / (b_ch + 1e-8)))
    result["rgb_gb_ratio_mean"] = float(np.mean(g_ch / (b_ch + 1e-8)))

    # ---- 8. Grey intensity skewness within lesion (asymmetry of brightness)
    gray_lesion = color.rgb2gray(image)[mask]
    result["gray_skew"] = float(skew(gray_lesion, bias=False)) if len(gray_lesion) > 2 else 0.0
    result["gray_kurtosis"] = float(
        np.mean((gray_lesion - gray_lesion.mean()) ** 4) / (gray_lesion.std() ** 4 + 1e-8)
    )

    return result


# ---------------------------------------------------------------------------
# Texture features (original + advanced)
# ---------------------------------------------------------------------------

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

    # Per-channel GLCM (R, G, B) for colour texture discrimination
    try:
        for ch_idx, ch_name in enumerate(["r", "g", "b"]):
            ch_full = image[:, :, ch_idx] if image.dtype == np.float32 else image[:, :, ch_idx].astype(np.float32) / 255.0
            ch_full = (ch_full * 255.0).astype(np.uint8)
            ch_crop, _ = crop_to_mask(ch_full, mask)
            ch_quantized = (ch_crop // 32).astype(np.uint8)
            ch_quantized = np.where(crop_mask, ch_quantized, 0).astype(np.uint8)
            ch_glcm = graycomatrix(
                ch_quantized,
                distances=[1, 2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=8,
                symmetric=True,
                normed=True,
            )
            result[f"glcm_{ch_name}_contrast_mean"] = float(np.mean(graycoprops(ch_glcm, "contrast")))
            result[f"glcm_{ch_name}_homogeneity_mean"] = float(np.mean(graycoprops(ch_glcm, "homogeneity")))
    except Exception:
        for ch_name in ["r", "g", "b"]:
            result[f"glcm_{ch_name}_contrast_mean"] = 0.0
            result[f"glcm_{ch_name}_homogeneity_mean"] = 0.0

    return result


def _advanced_texture_features(image, mask, feature_cfg):
    """Multi-scale LBP, local entropy, and border-zone GLCM."""
    gray = color.rgb2gray(image)
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

    result = {}

    # ---- 1. Local entropy (lesion region) --------------------------------
    try:
        ent_map = local_entropy(gray_uint8, disk(5))
        ent_lesion = ent_map[mask]
        result["local_entropy_mean"] = float(ent_lesion.mean())
        result["local_entropy_std"] = float(ent_lesion.std())
        # High-percentile entropy (presence of very chaotic areas)
        result["local_entropy_p90"] = float(np.percentile(ent_lesion, 90))
    except Exception:
        result["local_entropy_mean"] = 0.0
        result["local_entropy_std"] = 0.0
        result["local_entropy_p90"] = 0.0

    # ---- 2. Multi-radius LBP (capture texture at different scales) ----
    crop_gray, crop_mask = crop_to_mask(gray, mask)
    crop_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)

    radii = feature_cfg.get("lbp_multi_radii", [1, 5, 7])
    for r in radii:
        # Adjust points: keep approx (radius/3)*24 points — fewer for small radii
        pts = int(max(8, round(24 * r / 3)))
        try:
            lbp = local_binary_pattern(crop_uint8, pts, r, method="uniform")
            vals = lbp[crop_mask]
            if len(vals) == 0:
                continue
            bins = pts + 2
            hist, _ = np.histogram(vals, bins=bins, range=(0, bins), density=False)
            hist = hist.astype(np.float32)
            s = hist.sum()
            if s > 0:
                hist /= s
            for idx, v in enumerate(hist):
                result[f"lbp_r{r}_{idx}"] = float(v)
        except Exception:
            pass

    # ---- 3. Border-zone GLCM (texture at lesion edge) --------------------
    try:
        eroded = binary_erosion(mask, iterations=3)
        dilated = binary_dilation(mask, iterations=3)
        border_zone = dilated & ~eroded
        if border_zone.sum() >= 50:
            quantized = (crop_uint8 // 32).astype(np.uint8)
            bz = crop_mask & border_zone[crop_gray.shape[0]:, crop_gray.shape[1]:] if False else (
                # recompute in original image space for border zone
                None
            )
            # Simpler: compute border-zone in full image
            border_full = dilated & ~eroded
            if border_full.sum() >= 50:
                q_full = (gray_uint8 // 32).astype(np.uint8)
                q_border = np.where(border_full, q_full, 0).astype(np.uint8)
                glcm = graycomatrix(
                    q_border,
                    distances=[1, 2],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=8,
                    symmetric=True,
                    normed=True,
                )
                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
                    values = graycoprops(glcm, prop)
                    result[f"border_glcm_{prop}_mean"] = float(np.mean(values))
                    result[f"border_glcm_{prop}_std"] = float(np.std(values))
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Shape features (original + advanced)
# ---------------------------------------------------------------------------

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


def _advanced_shape_features(image, mask):
    """Fractal dimension, colour asymmetry, radial irregularity."""
    result = {}

    # ---- 1. Fractal dimension of the boundary ----------------------------
    result["fractal_dimension"] = _fractal_dimension(mask)

    # ---- 2. Colour-weighted asymmetry (MEL can be symmetric in shape
    #        but asymmetric in pigmentation) -------------------------------
    gray = color.rgb2gray(image)
    result["color_asymmetry_h"] = _color_asymmetry(gray, mask, axis=1)
    result["color_asymmetry_v"] = _color_asymmetry(gray, mask, axis=0)

    # ---- 3. Radial distance coefficient of variation --------------------
    result["radial_distance_cv"] = _radial_distance_cv(mask)

    # ---- 4. Border jaggedness: how many "indentations" along contour ----
    result["border_jaggedness"] = _border_jaggedness(mask)

    # ---- 5. Eccentricity-based asymmetry aspect ratio -------------------
    label_img = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(label_img)
    if regions:
        region = max(regions, key=lambda r: r.area)
        result["major_minor_ratio"] = float(
            region.major_axis_length / (region.minor_axis_length + 1e-8)
        )
    else:
        result["major_minor_ratio"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Clinical-inspired features (targeted at NV vs MEL)
# ---------------------------------------------------------------------------

def _clinical_features(image, mask):
    """Features derived from clinical dermoscopy criteria for melanoma."""
    from scipy.ndimage import label as nd_label

    result = {}

    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]
    gray = color.rgb2gray(image)

    # ---- 1. Distinct colour count (ABCDE: Colour variegation) --------------
    # Clinically meaningful colour categories in Lab space
    colors_present = 0
    # Black: L < 20
    if (lesion_lab[:, 0] < 20).mean() > 0.05:
        colors_present += 1
    # Dark brown: L 20-50, positive a
    dk_brown = (lesion_lab[:, 0] >= 20) & (lesion_lab[:, 0] < 50) & (lesion_lab[:, 1] > 2)
    if dk_brown.mean() > 0.05:
        colors_present += 1
    # Light brown: L 50-70
    lt_brown = (lesion_lab[:, 0] >= 50) & (lesion_lab[:, 0] < 70)
    if lt_brown.mean() > 0.05:
        colors_present += 1
    # Blue-gray: negative a and b
    blue_gray = (lesion_lab[:, 0] > 30) & (lesion_lab[:, 0] < 70) & (lesion_lab[:, 1] < -3) & (lesion_lab[:, 2] < -3)
    if blue_gray.mean() > 0.05:
        colors_present += 1
    # Red / pink: high a (>15)
    red = lesion_lab[:, 1] > 15
    if red.mean() > 0.05:
        colors_present += 1
    # White / scar-like: L > 80, low a and b
    white = (lesion_lab[:, 0] > 80) & (np.abs(lesion_lab[:, 1]) < 10) & (np.abs(lesion_lab[:, 2]) < 10)
    if white.mean() > 0.05:
        colors_present += 1

    result["clinical_color_count"] = float(colors_present)

    # Dominant colour class (which colour covers the largest area)
    class_sizes = [
        ("black", (lesion_lab[:, 0] < 20).mean()),
        ("dk_brown", dk_brown.mean()),
        ("lt_brown", lt_brown.mean()),
        ("blue_gray", blue_gray.mean()),
        ("red", red.mean()),
        ("white", white.mean()),
    ]
    if class_sizes:
        dominant = max(class_sizes, key=lambda x: x[1])
        result["clinical_dominant_color"] = float(
            ["black", "dk_brown", "lt_brown", "blue_gray", "red", "white"].index(dominant[0])
        )
        result["clinical_dominant_color_ratio"] = float(dominant[1])

    # ---- 2. Quadrant colour variance (asymmetric pigmentation) -------------
    ys, xs = np.where(mask)
    cy = float(np.median(ys)) if len(ys) > 0 else 0.0
    cx = float(np.median(xs)) if len(xs) > 0 else 0.0

    if len(ys) > 50:

        def _quadrant_mean(channel, y_lo, y_hi, x_lo, x_hi):
            """Mean of masked pixels within the given sub-region."""
            sub_mask = np.zeros_like(mask, dtype=bool)
            y0, y1 = max(0, y_lo), min(mask.shape[0], y_hi)
            x0, x1 = max(0, x_lo), min(mask.shape[1], x_hi)
            sub_mask[y0:y1, x0:x1] = True
            sub_mask &= mask
            if sub_mask.sum() < 10:
                return 0.0
            return float(channel[sub_mask].mean())

        for ch_idx, ch_name in enumerate(["l", "a", "b"]):
            ch = lab[:, :, ch_idx]
            # Quadrants: top-left, top-right, bottom-left, bottom-right
            tl = _quadrant_mean(ch, 0, int(cy), 0, int(cx))
            tr = _quadrant_mean(ch, 0, int(cy), int(cx), mask.shape[1])
            bl = _quadrant_mean(ch, int(cy), mask.shape[0], 0, int(cx))
            br = _quadrant_mean(ch, int(cy), mask.shape[0], int(cx), mask.shape[1])
            means = np.array([tl, tr, bl, br])
            result[f"clinical_quadrant_{ch_name}_var"] = float(np.var(means))
            result[f"clinical_quadrant_{ch_name}_range"] = float(np.ptp(means))

    # ---- 3. Dark-blob features -----------------------------------------
    try:
        # Use HSV Value channel (converted from Lab... let's use intensity instead)
        gray_lesion = gray[mask]
        dark_thresh = np.percentile(gray_lesion, 20) if len(gray_lesion) > 20 else 0.3
        dark_mask = (gray < dark_thresh) & mask
        if dark_mask.sum() > 10:
            struct = np.ones((3, 3), dtype=bool)
            labeled, n_blobs = nd_label(dark_mask, structure=struct)
            blob_sizes = np.bincount(labeled.ravel())[1:]  # exclude background

            result["clinical_dark_blob_count"] = float(n_blobs)
            result["clinical_dark_blob_mean_size"] = float(blob_sizes.mean()) if len(blob_sizes) > 0 else 0.0
            result["clinical_dark_blob_size_std"] = float(blob_sizes.std()) if len(blob_sizes) > 1 else 0.0
            # Spread: average distance of blobs from centroid
            if n_blobs > 1 and len(blob_sizes) > 1:
                blob_centroids = []
                for i in range(1, n_blobs + 1):
                    bys, bxs = np.where(labeled == i)
                    blob_centroids.append((bys.mean(), bxs.mean()))
                bc = np.array(blob_centroids)
                centroid_dist = np.sqrt((bc[:, 0] - cy)**2 + (bc[:, 1] - cx)**2)
                result["clinical_dark_blob_spread"] = float(centroid_dist.std())
            else:
                result["clinical_dark_blob_spread"] = 0.0
        else:
            result["clinical_dark_blob_count"] = 0.0
            result["clinical_dark_blob_mean_size"] = 0.0
            result["clinical_dark_blob_size_std"] = 0.0
            result["clinical_dark_blob_spread"] = 0.0
    except Exception:
        result["clinical_dark_blob_count"] = 0.0
        result["clinical_dark_blob_mean_size"] = 0.0
        result["clinical_dark_blob_size_std"] = 0.0
        result["clinical_dark_blob_spread"] = 0.0

    # ---- 4. Border gradient abruptness ------------------------------------
    try:
        eroded = binary_erosion(mask, iterations=2)
        border = mask & ~eroded
        bys, bxs = np.where(border)
        if len(bys) > 20:
            # For each border pixel, compute gradient magnitude
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gy**2 + gx**2)
            border_gradients = grad_mag[border]

            result["clinical_border_grad_mean"] = float(border_gradients.mean())
            result["clinical_border_grad_std"] = float(border_gradients.std())
            result["clinical_border_grad_p90"] = float(np.percentile(border_gradients, 90))
            result["clinical_border_grad_max"] = float(border_gradients.max())

            # Also compute gradient just OUTSIDE the lesion
            dilated = binary_dilation(mask, iterations=3)
            outer_rim = dilated & ~mask
            if outer_rim.sum() > 20:
                outer_grad = grad_mag[outer_rim]
                result["clinical_outer_grad_mean"] = float(outer_grad.mean())
                result["clinical_outer_grad_std"] = float(outer_grad.std())
            else:
                result["clinical_outer_grad_mean"] = 0.0
                result["clinical_outer_grad_std"] = 0.0
        else:
            for k in ["clinical_border_grad_mean", "clinical_border_grad_std",
                      "clinical_border_grad_p90", "clinical_border_grad_max",
                      "clinical_outer_grad_mean", "clinical_outer_grad_std"]:
                result[k] = 0.0
    except Exception:
        for k in ["clinical_border_grad_mean", "clinical_border_grad_std",
                  "clinical_border_grad_p90", "clinical_border_grad_max",
                  "clinical_outer_grad_mean", "clinical_outer_grad_std"]:
            result[k] = 0.0

    return result


# ---------------------------------------------------------------------------
# Melanin / Hemoglobin features (targeting MEL discrimination)
# ---------------------------------------------------------------------------

def _melanin_features(image, mask):
    """Melanin index, hemoglobin index, and peripheral pigmentation analysis."""
    result = {}
    lesion_rgb = image[mask]

    # ---- 1. Melanin index: -log(green) — melanin absorbs green strongly ----
    melanin = -np.log(np.maximum(lesion_rgb[:, 1], 1e-8))
    result["melanin_index_mean"] = float(melanin.mean())
    result["melanin_index_std"] = float(melanin.std())
    result["melanin_index_skew"] = (
        float(skew(melanin, bias=False)) if len(melanin) > 2 else 0.0
    )
    result["melanin_index_p90"] = float(np.percentile(melanin, 90))
    result["melanin_index_range"] = float(
        np.percentile(melanin, 95) - np.percentile(melanin, 5)
    )

    # ---- 2. Hemoglobin / erythema index: log(R/G) --------------------------
    hemo = np.log((lesion_rgb[:, 0] + 1e-8) / (lesion_rgb[:, 1] + 1e-8))
    result["hemoglobin_index_mean"] = float(hemo.mean())
    result["hemoglobin_index_std"] = float(hemo.std())
    result["hemoglobin_index_range"] = float(
        np.percentile(hemo, 95) - np.percentile(hemo, 5)
    )

    # ---- 3. Peripheral pigmentation features --------------------------------
    try:
        center = _center_region(mask, shrink_ratio=0.4)
        peripheral = mask & ~_center_region(mask, shrink_ratio=0.6)

        if peripheral.sum() > 50 and center.sum() > 50:
            melanin_map = -np.log(np.maximum(image[:, :, 1].astype(np.float64), 1e-8))
            mel_peri = melanin_map[peripheral].mean()
            mel_center = melanin_map[center].mean()
            result["peri_center_melanin_ratio"] = float(
                mel_peri / (mel_center + 1e-8)
            )

            gray = color.rgb2gray(image)
            dark_thresh = np.percentile(gray[mask], 30)
            peripheral_dark = (gray[peripheral] < dark_thresh).mean()
            center_dark = (gray[center] < dark_thresh).mean()
            result["peripheral_dark_ratio"] = float(peripheral_dark)
            result["center_dark_ratio"] = float(center_dark)
            result["peri_center_dark_ratio"] = float(
                peripheral_dark / (center_dark + 1e-8)
            )
        else:
            for k in [
                "peri_center_melanin_ratio", "peripheral_dark_ratio",
                "center_dark_ratio", "peri_center_dark_ratio",
            ]:
                result[k] = 0.0
    except Exception:
        for k in [
            "peri_center_melanin_ratio", "peripheral_dark_ratio",
            "center_dark_ratio", "peri_center_dark_ratio",
        ]:
            result[k] = 0.0

    # ---- 4. Melanin asymmetry (pigmentation asymmetry) ---------------------
    try:
        gray = color.rgb2gray(image)
        result["melanin_asymmetry_h"] = _color_asymmetry(gray, mask, axis=1)
        result["melanin_asymmetry_v"] = _color_asymmetry(gray, mask, axis=0)
    except Exception:
        result["melanin_asymmetry_h"] = 0.0
        result["melanin_asymmetry_v"] = 0.0

    return result


# ---------------------------------------------------------------------------
# HOG features (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dermoscopic colour features (ABCDE-C clinical 6-colour mapping)
# ---------------------------------------------------------------------------

# Oukil et al. 2021 RGB centroids (normalised Euclidean distance)
_OUKIL_CENTROIDS = {
    "light_brown":  (200, 155, 130),
    "medium_brown": (160, 100,  67),
    "dark_brown":   (126,  67,  48),
    "black":        ( 31,  26,  26),
    "blue_gray":    ( 75, 112, 137),
    "white":        (230, 230, 230),
}
# Normalisation factor: max possible Euclidean distance in RGB space
_NORM = np.sqrt(255.0**2 * 3)  # ~441.67


def _dermoscopic_color_features(image, mask):
    """6-colour clinical mapping (Oukil et al. 2021) + colour entropy."""
    result = {}
    lesion = image[mask].astype(np.float64)
    if len(lesion) < 10:
        for k in _dermo_color_defaults():
            result[k] = 0.0
        return result

    # Scale to [0, 255] to match Oukil centroids (image is [0, 1] after preprocessing)
    lesion_255 = lesion * 255.0

    centroids = np.array(list(_OUKIL_CENTROIDS.values()), dtype=np.float64)
    centroid_colors = list(_OUKIL_CENTROIDS.keys())

    dists = np.sqrt(((lesion_255[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    dists /= _NORM
    nearest = np.argmin(dists, axis=1)

    area_ratios = {}
    for i, cname in enumerate(centroid_colors):
        ratio = (nearest == i).mean()
        result[f"dermo_{cname}_ratio"] = float(ratio)
        area_ratios[cname] = ratio

    # Number of colours covering >= 5% of lesion area
    present = sum(1 for r in area_ratios.values() if r >= 0.05)
    result["dermo_color_count"] = float(present)

    # Dominant colour
    if area_ratios:
        dominant = max(area_ratios, key=area_ratios.get)
        result["dermo_dominant_color_ratio"] = float(area_ratios[dominant])

    # Colour Shannon entropy
    counts = np.bincount(nearest, minlength=len(centroid_colors))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    result["dermo_color_entropy"] = float(-np.sum(probs * np.log2(probs)))

    # Blue-white combined ratio (blue_gray + white)
    result["dermo_blue_white_ratio"] = float(area_ratios.get("blue_gray", 0) + area_ratios.get("white", 0))

    # Colour spatial dispersion: std of distances from each colour's centroid to lesion centre
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    color_dispersions = []
    for i in range(len(centroid_colors)):
        c_pixels = np.where(nearest == i)[0]
        if len(c_pixels) > 10:
            cy_i = ys[c_pixels].mean()
            cx_i = xs[c_pixels].mean()
            color_dispersions.append(np.sqrt((cy_i - cy)**2 + (cx_i - cx)**2))
    result["dermo_color_dispersion_std"] = float(np.std(color_dispersions)) if len(color_dispersions) > 1 else 0.0

    return result


def _dermo_color_defaults():
    defaults = []
    for c in _OUKIL_CENTROIDS:
        defaults.append(f"dermo_{c}_ratio")
    defaults += ["dermo_color_count", "dermo_dominant_color_ratio", "dermo_color_entropy",
                  "dermo_blue_white_ratio", "dermo_color_dispersion_std"]
    return defaults


# ---------------------------------------------------------------------------
# Dermoscopic asymmetry features (ABCDE-A / TDS A-score)
# ---------------------------------------------------------------------------

def _dermoscopic_asymmetry_features(image, mask):
    """TDS-style shape + pigment asymmetry (Pellacani et al. 2004)."""
    result = {}
    try:
        from skimage.measure import regionprops

        regions = regionprops(mask.astype(np.uint8))
        if not regions:
            return _asymmetry_defaults()
        region = max(regions, key=lambda r: r.area)
        orientation = region.orientation
        cy, cx = region.centroid

        # ---- Shape asymmetry along major axis ----
        result["tds_asymmetry_major_shape"] = _reflect_asymmetry(mask, cy, cx, orientation)
        result["tds_asymmetry_minor_shape"] = _reflect_asymmetry(mask, cy, cx, orientation + np.pi / 2)

        # ---- Pigment asymmetry (using dark area from median threshold) ----
        gray = color.rgb2gray(image)
        dark_thresh = np.median(gray[mask]) if mask.any() else 0.5
        dark_mask = (gray < dark_thresh) & mask
        result["tds_asymmetry_major_pigment"] = _reflect_asymmetry(dark_mask, cy, cx, orientation)
        result["tds_asymmetry_minor_pigment"] = _reflect_asymmetry(dark_mask, cy, cx, orientation + np.pi / 2)

        # TDS A score (0, 1, or 2)
        shape_asym = (result["tds_asymmetry_major_shape"] < 0.90) + (result["tds_asymmetry_minor_shape"] < 0.90)
        pigment_asym = (result["tds_asymmetry_major_pigment"] < 0.80) + (result["tds_asymmetry_minor_pigment"] < 0.80)
        result["tds_a_score"] = float(np.clip(max(shape_asym, pigment_asym), 0, 2))

        # ---- Quadrant colour variance ratio ----
        h, w = mask.shape
        quadrant_means = []
        for y_lo, y_hi in [(0, int(cy)), (int(cy), h)]:
            for x_lo, x_hi in [(0, int(cx)), (int(cx), w)]:
                q_mask = np.zeros_like(mask)
                q_mask[y_lo:y_hi, x_lo:x_hi] = True
                q_mask &= mask
                if q_mask.sum() > 10:
                    quadrant_means.append(image[q_mask].mean(axis=0))
        if len(quadrant_means) >= 4:
            qm = np.array(quadrant_means)
            within_var = qm.var(axis=0).mean()
            total_var = image[mask].var(axis=0).mean()
            result["tds_quadrant_color_var_ratio"] = float(within_var / (total_var + 1e-8))
        else:
            result["tds_quadrant_color_var_ratio"] = 0.0

    except Exception:
        result.update(_asymmetry_defaults())
    return result


def _asymmetry_defaults():
    return {
        "tds_asymmetry_major_shape": 1.0, "tds_asymmetry_minor_shape": 1.0,
        "tds_asymmetry_major_pigment": 1.0, "tds_asymmetry_minor_pigment": 1.0,
        "tds_a_score": 0.0, "tds_quadrant_color_var_ratio": 0.0,
    }


def _reflect_asymmetry(mask_2d, cy, cx, angle_rad):
    """Return overlap ratio (IoU) after reflecting one half across an axis."""
    h, w = mask_2d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # Signed distance from axis
    signed_dist = (xx - cx) * np.sin(angle_rad) - (yy - cy) * np.cos(angle_rad)
    half1 = (signed_dist >= 0) & mask_2d
    half2 = (signed_dist < 0) & mask_2d

    # Reflect half2 across the axis
    perp_x = -np.cos(angle_rad)
    perp_y = -np.sin(angle_rad)
    proj = (yy - cy) * perp_y + (xx - cx) * perp_x
    reflect_y = (yy.astype(float) - 2 * proj * perp_y).round().astype(int)
    reflect_x = (xx.astype(float) - 2 * proj * perp_x).round().astype(int)
    valid = (reflect_y >= 0) & (reflect_y < h) & (reflect_x >= 0) & (reflect_x < w)
    reflected = np.zeros_like(mask_2d)
    reflected[reflect_y[valid], reflect_x[valid]] = half2[valid]

    intersection = (half1 & reflected).sum()
    union = (half1 | reflected).sum()
    return float(intersection / union) if union > 0 else 1.0


# ---------------------------------------------------------------------------
# Dermoscopic border features (ABCDE-B)
# ---------------------------------------------------------------------------

def _dermoscopic_border_features(image, mask):
    """Notch count, 8-octant abruptness, enhanced fractal dimension, gradient CV."""
    result = {}
    try:
        eroded = binary_erosion(mask, iterations=2)
        boundary = mask & ~eroded
        bys, bxs = np.where(boundary)

        if len(bys) < 20:
            return _border_defaults()

        # ---- 1. Notch / indentation count (Jaworek-Korjakowska 2015) ----
        cy, cx = bys.mean(), bxs.mean()
        angles = np.arctan2(bys - cy, bxs - cx)
        distances = np.sqrt((bys - cy)**2 + (bxs - cx)**2)
        order = np.argsort(angles)
        r_sorted = distances[order]
        sigma = max(2, len(r_sorted) // 30)
        sm_sorted = gaussian_filter1d(r_sorted.astype(float), sigma)
        deriv = np.diff(sm_sorted)
        minima_mask = (np.diff(np.sign(deriv)) > 0)
        minima_idx = np.where(minima_mask)[0] + 1
        if len(minima_idx) > 0:
            mean_r = sm_sorted.mean()
            deep_notches = [i for i in minima_idx if sm_sorted[i] < mean_r * 0.9]
            result["border_notch_count"] = float(len(deep_notches))
        else:
            result["border_notch_count"] = 0.0

        # ---- 2. 8-octant border abruptness score (TDS B) ----
        gray = color.rgb2gray(image)
        octant_score = 0
        for octant in range(8):
            angle_lo = octant * np.pi / 4 - np.pi
            angle_hi = (octant + 1) * np.pi / 4 - np.pi
            # Outer rim pixels for this octant
            dilated = binary_dilation(mask, iterations=8)
            outer_rim = dilated & ~mask
            or_ys, or_xs = np.where(outer_rim)
            if len(or_ys) < 10:
                continue
            o_angles = np.arctan2(or_ys - cy, or_xs - cx)
            in_octant = (o_angles >= angle_lo) & (o_angles < angle_hi)
            if in_octant.sum() < 10:
                continue
            grad_y, grad_x = np.gradient(gray)
            grad_mag = np.sqrt(grad_y**2 + grad_x**2)
            octant_grad = grad_mag[outer_rim][in_octant] if in_octant.sum() <= len(or_ys) else np.zeros(1)
            if len(octant_grad) > 0 and np.mean(octant_grad) > 0.03:
                octant_score += 1
        result["border_abruptness_score"] = float(octant_score)  # 0-8

        # ---- 3. Enhanced fractal dimension ----
        result["border_fractal_dimension_enhanced"] = _fractal_dimension_enhanced(mask)

        # ---- 4. Border gradient coefficient of variation ----
        grad_y, grad_x = np.gradient(gray)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        border_grads = grad_mag[boundary]
        if len(border_grads) > 0:
            result["border_grad_cv"] = float(border_grads.std() / (border_grads.mean() + 1e-8))
        else:
            result["border_grad_cv"] = 0.0

    except Exception:
        result.update(_border_defaults())
    return result


def _border_defaults():
    return {"border_notch_count": 0.0, "border_abruptness_score": 0.0,
            "border_fractal_dimension_enhanced": 0.0, "border_grad_cv": 0.0}



def _fractal_dimension_enhanced(mask):
    """Box-counting with more sizes for better FD estimate."""
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    if boundary.sum() < 20:
        return 0.0
    ys, xs = np.where(boundary)
    points = np.column_stack([ys, xs])
    sizes = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    counts = []
    for s in sizes:
        boxes = set()
        stride = max(1, len(points) // 3000)
        for p in points[::stride]:
            boxes.add((p[0] // s, p[1] // s))
        if len(boxes) > 0:
            counts.append(len(boxes))
    if len(counts) < 4:
        return 0.0
    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
    return float(-coeffs[0])


# ---------------------------------------------------------------------------
# Pigment network features (7-point checklist — major criterion)
# ---------------------------------------------------------------------------

def _pigment_network_features(image, mask):
    """Gabor-based pigment network regularity analysis."""
    result = {}
    try:
        gray = color.rgb2gray(image)
        orientation_values = np.linspace(0, np.pi, 9)[:8]  # 8 orientations

        energies = []
        full_mags = []
        for theta in orientation_values:
            for freq in (0.1, 0.2):
                try:
                    real, imag = filters.gabor(gray, frequency=freq, theta=theta)
                    mag = np.sqrt(real**2 + imag**2)
                    energies.append(mag[mask].mean())
                    full_mags.append(mag)
                except Exception:
                    pass

        if energies:
            result["pigment_network_gabor_energy_mean"] = float(np.mean(energies))
            result["pigment_network_gabor_energy_std"] = float(np.std(energies))

            # Energy entropy: regular network has energy concentrated in few orientations
            if len(energies) >= 4:
                e_arr = np.array(energies)
                e_sum = e_arr.sum()
                if e_sum > 0:
                    e_prob = e_arr / e_sum
                    e_prob = e_prob[e_prob > 0]
                    result["pigment_network_gabor_entropy"] = float(-np.sum(e_prob * np.log2(e_prob)))
                else:
                    result["pigment_network_gabor_entropy"] = 0.0
            else:
                result["pigment_network_gabor_entropy"] = 0.0

            # Reticular area ratio: pixels with strong Gabor response
            max_mag = np.maximum.reduce(full_mags)
            if max_mag.max() > 0:
                thresh = filters.threshold_otsu(max_mag[mask]) if mask.any() else 0.1
                reticular = (max_mag > thresh) & mask
                result["pigment_network_reticular_ratio"] = float(reticular.sum() / mask.sum())
            else:
                result["pigment_network_reticular_ratio"] = 0.0
        else:
            result["pigment_network_gabor_energy_mean"] = 0.0
            result["pigment_network_gabor_energy_std"] = 0.0
            result["pigment_network_gabor_entropy"] = 0.0
            result["pigment_network_reticular_ratio"] = 0.0

        # Dark reticular line density (morphological black-hat + skeleton)
        try:
            gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            crop_img, crop_mask = crop_to_mask(gray_uint8, mask)
            se = morphology.disk(3)
            blackhat = morphology.black_tophat(crop_img.astype(float), se)
            # Threshold to get dark lines
            dark_lines = (blackhat > 8) & crop_mask
            if dark_lines.sum() > 10:
                skeleton = morphology.skeletonize(dark_lines)
                result["pigment_network_line_density"] = float(skeleton.sum() / crop_mask.sum())
            else:
                result["pigment_network_line_density"] = 0.0
        except Exception:
            result["pigment_network_line_density"] = 0.0

    except Exception:
        for k in ["pigment_network_gabor_energy_mean", "pigment_network_gabor_energy_std",
                  "pigment_network_gabor_entropy", "pigment_network_reticular_ratio",
                  "pigment_network_line_density"]:
            result[k] = 0.0
    return result



# ---------------------------------------------------------------------------
# Dots / globules features (7-point checklist — minor criterion)
# ---------------------------------------------------------------------------

def _dots_globules_features(image, mask):
    """LoG blob detection for atypical dots and globules."""
    result = {}
    try:
        gray = color.rgb2gray(image)
        # Detect dark blobs inside lesion
        blobs = blob_log(gray * mask.astype(float), min_sigma=1.0, max_sigma=5.0,
                         num_sigma=10, threshold=0.03)
        if len(blobs) == 0:
            return _dots_defaults()

        # Filter: centre of blob must be within mask
        cy, cx = blobs[:, 0].astype(int), blobs[:, 1].astype(int)
        valid = mask[cy.clip(0, mask.shape[0]-1), cx.clip(0, mask.shape[1]-1)]
        blobs = blobs[valid]
        if len(blobs) == 0:
            return _dots_defaults()

        result["dots_globules_count"] = float(len(blobs))
        sigmas = blobs[:, 2]
        result["dots_globules_sigma_mean"] = float(sigmas.mean())
        result["dots_globules_sigma_std"] = float(sigmas.std()) if len(sigmas) > 1 else 0.0

        # Nearest-neighbour distance variability (irregular spacing = melanoma)
        positions = blobs[:, :2]
        if len(positions) > 2:
            tree = KDTree(positions)
            dists, _ = tree.query(positions, k=2)
            nn_dists = dists[:, 1]  # nearest neighbour distance
            result["dots_globules_nn_dist_mean"] = float(nn_dists.mean())
            result["dots_globules_nn_dist_cv"] = float(nn_dists.std() / (nn_dists.mean() + 1e-8))
        else:
            result["dots_globules_nn_dist_mean"] = 0.0
            result["dots_globules_nn_dist_cv"] = 0.0

        # Peripheral density ratio (peripheral 30% ring vs centre)
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()
        max_dist = np.sqrt((ys - cy)**2 + (xs - cx)**2).max()
        blob_dists = np.sqrt((blobs[:, 0] - cy)**2 + (blobs[:, 1] - cx)**2)
        peripheral_mask = blob_dists > 0.7 * max_dist
        centre_mask = blob_dists <= 0.7 * max_dist
        peri_density = peripheral_mask.sum() / max(mask[ys > 0].sum() * 0.3, 1)
        centre_density = (centre_mask.sum() / max(mask.sum() * 0.7, 1)) if centre_mask.sum() > 0 else 0
        result["dots_peripheral_density_ratio"] = float(
            peri_density / (centre_density + 1e-8)) if centre_density > 0 else 1.0

    except Exception:
        result.update(_dots_defaults())
    return result


def _dots_defaults():
    return {
        "dots_globules_count": 0.0, "dots_globules_sigma_mean": 0.0,
        "dots_globules_sigma_std": 0.0, "dots_globules_nn_dist_mean": 0.0,
        "dots_globules_nn_dist_cv": 0.0, "dots_peripheral_density_ratio": 0.0,
    }


# ---------------------------------------------------------------------------
# Streak / pseudopod features (7-point checklist — minor criterion)
# ---------------------------------------------------------------------------

def _streak_features(image, mask):
    """Radial streak analysis at lesion periphery."""
    result = {}
    try:
        gray = color.rgb2gray(image)

        # Define peripheral ring: 10%-30% of lesion radius from boundary
        eroded_inner = binary_erosion(mask, iterations=int(mask.shape[0] * 0.03))
        if not eroded_inner.any():
            return _streak_defaults()

        peripheral = mask & ~eroded_inner
        if peripheral.sum() < 50:
            return _streak_defaults()

        # ---- 1. Radial gradient consistency ----
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()
        per_ys, per_xs = np.where(peripheral)

        radial_angles = np.arctan2(per_ys - cy, per_xs - cx)
        grad_y, grad_x = np.gradient(gray)
        grad_angles = np.arctan2(grad_y[peripheral], grad_x[peripheral])
        grad_mag = np.sqrt(grad_y[peripheral]**2 + grad_x[peripheral]**2)

        # Cosine similarity between gradient direction and radial direction
        angle_diff = np.cos(radial_angles - grad_angles)  # 1 = aligned, -1 = opposite
        # Weight by gradient magnitude
        if grad_mag.sum() > 0:
            alignment = (angle_diff * grad_mag).sum() / grad_mag.sum()
            result["streak_radial_alignment"] = float(alignment)
        else:
            result["streak_radial_alignment"] = 0.0

        # Angle concentration (high concentration = streaks pointing in few directions)
        strong_grad = grad_mag > np.percentile(grad_mag, 75)
        if strong_grad.sum() > 10:
            strong_angles = radial_angles[strong_grad]
            # Circular variance (1 - |mean of unit vectors|)
            mean_vec = np.array([np.cos(strong_angles).mean(), np.sin(strong_angles).mean()])
            circ_var = 1.0 - np.sqrt((mean_vec**2).sum())
            result["streak_angle_concentration"] = float(1.0 - circ_var)  # high = concentrated
        else:
            result["streak_angle_concentration"] = 0.0

        # ---- 2. Peripheral FFT high-frequency energy (boundary irregularities) ----
        eroded = binary_erosion(mask, iterations=2)
        boundary = mask & ~eroded
        bys, bxs = np.where(boundary)
        if len(bys) > 20:
            angles = np.arctan2(bys - cy, bxs - cx)
            r_vals = np.sqrt((bys - cy)**2 + (bxs - cx)**2)
            order = np.argsort(angles)
            r_sorted = r_vals[order]
            # FFT of radial function
            fft = np.abs(np.fft.rfft(r_sorted))
            total = fft.sum()
            if total > 0:
                # High-frequency energy ratio (top 50% of frequencies)
                mid = len(fft) // 2
                result["streak_border_hf_energy"] = float(fft[mid:].sum() / total)
            else:
                result["streak_border_hf_energy"] = 0.0
        else:
            result["streak_border_hf_energy"] = 0.0

    except Exception:
        result.update(_streak_defaults())
    return result


def _streak_defaults():
    return {"streak_radial_alignment": 0.0, "streak_angle_concentration": 0.0,
            "streak_border_hf_energy": 0.0}


# ---------------------------------------------------------------------------
# Regression structures / blue-white veil (7-point checklist — enhanced)
# ---------------------------------------------------------------------------

def _regression_structures_features(image, mask):
    """Enhanced blue-white veil + regression structure detection."""
    result = {}
    try:
        lab = color.rgb2lab(image)
        hsv = color.rgb2hsv(image)
        lesion_lab = lab[mask]
        lesion_hsv = hsv[mask]
        if len(lesion_lab) < 10:
            return _regression_defaults()

        # ---- 1. Blue-white veil (enhanced — Celebi et al. 2008 inspired) ----
        # Blue-gray: B-channel dominant, medium-low intensity, low saturation
        r, g, b = image[:, :, 0].astype(float), image[:, :, 1].astype(float), image[:, :, 2].astype(float)
        # Blue dominance
        blue_dominant = (b > r) & (b > g)
        # Medium intensity (not too dark, not too bright)
        intensity = (r + g + b) / 3.0
        medium_intensity = (intensity > 40) & (intensity < 180)
        # Low saturation
        low_saturation = hsv[:, :, 1] < 0.4

        bwv_mask = blue_dominant & medium_intensity & low_saturation & mask
        result["blue_white_veil_enhanced_ratio"] = float(bwv_mask.sum() / mask.sum())

        # ---- 2. White regression areas (scar-like depigmentation) ----
        white_mask = (
            (lab[:, :, 0] > 85) &
            (np.abs(lab[:, :, 1]) < 5) &
            (np.abs(lab[:, :, 2]) < 5) &
            mask
        )
        result["regression_white_ratio"] = float(white_mask.sum() / mask.sum())

        # ---- 3. Blue-gray peppering (small pepper-like granules in regression) ----
        blue_gray_mask = (
            (lesion_lab[:, 1] < -3) &
            (lesion_lab[:, 2] < -3) &
            (lesion_lab[:, 0] > 30) &
            (lesion_lab[:, 0] < 70)
        )
        result["regression_pepper_ratio"] = float(blue_gray_mask.mean())

    except Exception:
        result.update(_regression_defaults())
    return result


def _regression_defaults():
    return {"blue_white_veil_enhanced_ratio": 0.0, "regression_white_ratio": 0.0,
            "regression_pepper_ratio": 0.0}


def gray_scaled(image):
    """Return float gray image in [0, 1]."""
    return color.rgb2gray(image) if image.ndim == 3 else image.astype(float) / 255.0


# ---------------------------------------------------------------------------
# Composite ratio features (targeting large-but-benign NV vs MEL)
# Key insight: wrong NV shares MEL-like size but retains NV-like colours.
# Ratios of colour ÷ size amplify this separation signal.
# ---------------------------------------------------------------------------

def _composite_features(image, mask):
    """Composite features: colour × shape ratios to separate large NV from MEL."""
    result = {}
    try:
        lesion = image[mask].astype(np.float64)
        if len(lesion) < 10:
            return _composite_defaults()

        eps = 1e-8
        h, w = mask.shape

        # ---- Size & shape proxies ----
        area_r = mask.sum() / mask.size
        perimeter_val = float(measure.perimeter(mask))
        compactness = float(perimeter_val / (np.sqrt(mask.sum()) + eps))

        # Asymmetry (horizontal + vertical)
        from skimage.measure import regionprops
        regions = regionprops(mask.astype(np.uint8))
        if regions:
            r = max(regions, key=lambda x: x.area)
            horiz_asym = float(_asymmetry(mask, axis=1))
            vert_asym = float(_asymmetry(mask, axis=0))
            eccentricity = float(r.eccentricity)
            solidity = float(r.solidity)
            circularity = float(4.0 * np.pi * r.area / (perimeter_val**2 + eps))
        else:
            horiz_asym = vert_asym = eccentricity = solidity = circularity = 1.0

        # ---- Colour features that distinguish NV from MEL ----
        hsv = color.rgb2hsv(image)
        lesion_hsv = hsv[mask]
        hue_mean = float(lesion_hsv[:, 0].mean())
        hue_std = float(lesion_hsv[:, 0].std())
        sat_mean = float(lesion_hsv[:, 1].mean())
        val_mean = float(lesion_hsv[:, 2].mean())

        lab = color.rgb2lab(image)
        lesion_lab = lab[mask]
        lab_b_mean = float(lesion_lab[:, 2].mean())

        # ---- Colour ratios (NV has distinct colour signatures) ----
        r_ch, g_ch, b_ch = lesion[:, 0], lesion[:, 1], lesion[:, 2]
        gb_ratio = float(np.mean(g_ch / (b_ch + eps)))  # NV has lower G/B ratio than MEL

        # ---- Dark pixel ratio ----
        dark_ratio = float(np.mean(lesion_hsv[:, 2] < 0.35))

        # ---- Composite features: colour ÷ size ----
        # Core idea: large+benign has NV-like colour profile; large+malignant has MEL-like
        result["composite_gb_ratio_per_size"] = float(gb_ratio / (area_r + eps))
        result["composite_hue_mean_per_size"] = float(hue_mean / (area_r + eps))
        result["composite_dark_per_size"] = float(dark_ratio / (area_r + eps))
        result["composite_lab_b_per_size"] = float(abs(lab_b_mean) / (area_r + eps))

        # Shape × colour interactions
        result["composite_solidity_per_size"] = float(solidity / (area_r + eps))
        result["composite_circularity_per_size"] = float(circularity / (area_r + eps))
        result["composite_compactness_x_size"] = float(compactness * area_r)
        result["composite_asymmetry_per_size"] = float((horiz_asym + vert_asym) / (area_r + eps))

        # Colour × shape interaction (large lesion with uniform colour → benign)
        result["composite_hue_std_x_asymmetry"] = float(hue_std * (horiz_asym + vert_asym))
        result["composite_sat_x_eccentricity"] = float(sat_mean * eccentricity)

        # Border complexity relative to size (large+simple → benign, large+complex → malignant)
        border_irreg = float(perimeter_val / (measure.perimeter(regions[0].convex_image) + eps)) if regions else 1.0
        result["composite_border_per_size"] = float(border_irreg / (area_r + eps))

    except Exception:
        result.update(_composite_defaults())
    return result


def _composite_defaults():
    return {
        "composite_gb_ratio_per_size": 0.0,
        "composite_hue_mean_per_size": 0.0,
        "composite_dark_per_size": 0.0,
        "composite_lab_b_per_size": 0.0,
        "composite_solidity_per_size": 0.0,
        "composite_circularity_per_size": 0.0,
        "composite_compactness_x_size": 0.0,
        "composite_asymmetry_per_size": 0.0,
        "composite_hue_std_x_asymmetry": 0.0,
        "composite_sat_x_eccentricity": 0.0,
        "composite_border_per_size": 0.0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_std(image, mask, window_size=9):
    """Mean of the local standard deviation within the masked region."""
    mean = uniform_filter(image, size=window_size)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    var = np.maximum(mean_sq - mean ** 2, 0)
    local_std = np.sqrt(var)
    return float(local_std[mask].mean())


def _center_region(mask, shrink_ratio=0.5):
    """Return eroded mask representing the central portion."""
    label_img = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(label_img)
    if not regions:
        return mask.astype(bool)
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    cy, cx = (minr + maxr) / 2.0, (minc + maxc) / 2.0
    half_h = (maxr - minr) * shrink_ratio / 2.0
    half_w = (maxc - minc) * shrink_ratio / 2.0
    center = np.zeros_like(mask, dtype=bool)
    y0 = max(int(round(cy - half_h)), 0)
    y1 = min(int(round(cy + half_h)), mask.shape[0])
    x0 = max(int(round(cx - half_w)), 0)
    x1 = min(int(round(cx + half_w)), mask.shape[1])
    center[y0:y1, x0:x1] = True
    return center & mask


def _fractal_dimension(mask):
    """Box-counting fractal dimension of the lesion boundary."""
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    if boundary.sum() < 20:
        return 0.0

    ys, xs = np.where(boundary)
    points = np.column_stack([ys, xs])

    sizes = np.array([2, 4, 8, 16, 32, 64])
    counts = []
    for s in sizes:
        boxes = set()
        stride = max(1, len(points) // 3000)
        for p in points[::stride]:
            boxes.add((p[0] // s, p[1] // s))
        if len(boxes) > 0:
            counts.append(len(boxes))

    if len(counts) < 3:
        return 0.0

    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
    return float(-coeffs[0])


def _color_asymmetry(image_channel, mask, axis):
    """Asymmetry index weighted by pixel intensity (colour asymmetry)."""
    flipped_mask = np.flip(mask, axis=axis)
    flipped_intensity = np.flip(image_channel, axis=axis)

    overlap = mask & flipped_mask
    if overlap.sum() == 0:
        return 0.0

    diff = np.abs(image_channel * mask - flipped_intensity * flipped_mask)
    return float(diff[overlap].mean())


def _radial_distance_cv(mask):
    """Coefficient of variation of radial distances: centroid → boundary."""
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return 0.0
    cy, cx = ys.mean(), xs.mean()

    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    bys, bxs = np.where(boundary)
    if len(bys) < 10:
        return 0.0

    distances = np.sqrt((bys - cy) ** 2 + (bxs - cx) ** 2)
    m = distances.mean()
    return float(distances.std() / m) if m > 0 else 0.0


def _border_jaggedness(mask):
    """Number of significant concavity / convexity changes along the border.

    A simple proxy: count how often the radial distance changes direction.
    """
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return 0.0
    cy, cx = ys.mean(), xs.mean()

    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    bys, bxs = np.where(boundary)
    if len(bys) < 20:
        return 0.0

    angles = np.arctan2(bys - cy, bxs - cx)
    distances = np.sqrt((bys - cy) ** 2 + (bxs - cx) ** 2)

    # Sort by angle
    order = np.argsort(angles)
    sorted_dist = distances[order]

    # Smooth with a moving average
    window = max(3, len(sorted_dist) // 20)
    kernel = np.ones(window) / window
    smoothed = np.convolve(sorted_dist, kernel, mode="same")

    # Count sign changes in derivative (= inflection points)
    deriv = np.diff(smoothed)
    sign_changes = np.sum(np.diff(np.sign(deriv)) != 0)
    return float(sign_changes)
