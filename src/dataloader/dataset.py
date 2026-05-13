from pathlib import Path

import pandas as pd
from PIL import Image


def parse_base_id(image_id):
    image_id = str(image_id)
    return image_id.replace("_aug1", "").replace("_aug2", "")


def parse_augmentation_id(image_id):
    image_id = str(image_id)
    if image_id.endswith("_aug1"):
        return "aug1"
    if image_id.endswith("_aug2"):
        return "aug2"
    return "original"


def build_metadata(config):
    data_config = config["data"]
    image_dir = Path(data_config["image_dir"])
    mask_dir = Path(data_config["mask_dir"])
    label_csv = Path(data_config["label_csv"])

    df = pd.read_csv(label_csv)
    if not {"image_id", "dx"}.issubset(df.columns):
        raise ValueError("label.csv must contain image_id and dx columns.")

    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["dx"].astype(str)
    df["base_id"] = df["image_id"].apply(parse_base_id)
    df["augmentation_id"] = df["image_id"].apply(parse_augmentation_id)
    df["is_augmented"] = df["augmentation_id"] != "original"
    df["image_path"] = df["image_id"].apply(lambda x: str(image_dir / f"{x}.jpg"))
    df["mask_path"] = df["image_id"].apply(lambda x: str(mask_dir / f"mask_{x}.jpg"))
    return df[
        [
            "image_id",
            "label",
            "image_path",
            "mask_path",
            "base_id",
            "is_augmented",
            "augmentation_id",
        ]
    ]


def validate_metadata(df, strict_groups=True):
    errors = []

    for row in df.itertuples(index=False):
        image_path = Path(row.image_path)
        mask_path = Path(row.mask_path)
        if not image_path.exists():
            errors.append(f"Missing image: {image_path}")
            continue
        if not mask_path.exists():
            errors.append(f"Missing mask: {mask_path}")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                errors.append(
                    f"Image/mask size mismatch for {row.image_id}: "
                    f"{image.size} vs {mask.size}"
                )

    if strict_groups:
        expected = {"original", "aug1", "aug2"}
        for base_id, group in df.groupby("base_id"):
            actual = set(group["augmentation_id"])
            if actual != expected:
                errors.append(f"base_id={base_id} has augmentations {sorted(actual)}")

    if errors:
        sample = "\n".join(errors[:20])
        raise ValueError(f"Metadata validation failed with {len(errors)} errors:\n{sample}")

    return True
