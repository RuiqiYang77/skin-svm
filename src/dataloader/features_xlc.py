import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "dip_project_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation, uniform_filter
from scipy.stats import skew
from skimage import color, measure
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
from skimage.transform import resize
from tqdm import tqdm

from src.dataloader.preprocessing import crop_to_mask, load_image_and_mask


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features(image_path, mask_path, config):
    image, mask = load_image_and_mask(image_path, mask_path, config)
    features = {}
    feature_cfg = config["svm"]["features"]

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
