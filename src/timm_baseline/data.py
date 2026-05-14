"""timm 对比基线的数据集和预处理工具。

用途：
    复用现有项目的元数据和 mask 路径，为深度学习对比基线构建 PyTorch 数据集和加载器。

主要入口：
    build_split_loaders(...)
    prepare_single_tensor(...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dataloader.preprocessing import crop_to_mask


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def build_class_mapping(metadata_df: pd.DataFrame):
    # 保持类别顺序确定，这样 checkpoint 和报告都用同一套标签->索引映射。
    class_names = sorted(metadata_df["label"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    return class_names, label_to_idx


def build_transforms(img_size: int, train: bool, normalize: bool = True):
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomRotation(15),
            ]
        )
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    return transforms.Compose(steps)


def _prepare_pil_image(
    image_path: str,
    mask_path: str | None,
    use_mask_crop: bool,
    mask_threshold: int,
    mask_padding: int,
):
    image = Image.open(image_path).convert("RGB")
    if not use_mask_crop or not mask_path:
        return image

    # 可选的病灶 mask 裁剪，保持基线和原来病灶聚焦的设计思路一致。
    mask = np.array(Image.open(mask_path).convert("L"))
    binary_mask = mask > mask_threshold
    if not binary_mask.any():
        raise ValueError(f"Empty lesion mask: {mask_path}")

    image_array = np.array(image)
    crop_image, _ = crop_to_mask(image_array, binary_mask, padding=mask_padding)
    if crop_image.size == 0:
        return image
    return Image.fromarray(crop_image.astype(np.uint8))


class LesionClassificationDataset(Dataset):
    # 轻量级的 ImageFolder 风格包装，同时保留元数据供后面生成报告。
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        label_to_idx: dict[str, int],
        img_size: int,
        train: bool,
        use_mask_crop: bool,
        mask_threshold: int,
        mask_padding: int,
        normalize: bool = True,
    ):
        self.records = metadata_df.reset_index(drop=True).to_dict("records")
        self.label_to_idx = label_to_idx
        self.use_mask_crop = use_mask_crop
        self.mask_threshold = mask_threshold
        self.mask_padding = mask_padding
        self.transform = build_transforms(img_size, train=train, normalize=normalize)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = _prepare_pil_image(
            record["image_path"],
            record["mask_path"],
            self.use_mask_crop,
            self.mask_threshold,
            self.mask_padding,
        )
        image_tensor = self.transform(image)
        label_name = str(record["label"])
        label_idx = int(self.label_to_idx[label_name])
        meta = {
            "image_id": record["image_id"],
            "label": label_name,
            "label_idx": label_idx,
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
            "base_id": record["base_id"],
            "augmentation_id": record["augmentation_id"],
        }
        return image_tensor, label_idx, meta


def build_split_loaders(metadata_df: pd.DataFrame, split_df: pd.DataFrame, config):
    timm_cfg = config["timm"]
    # 先合并元数据和 split 标签一次，然后为每个 split 各建一个数据集。
    merged = metadata_df.merge(
        split_df[["image_id", "split"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    class_names, label_to_idx = build_class_mapping(merged)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        split_part = merged[merged["split"] == split_name].copy()
        dataset = LesionClassificationDataset(
            split_part,
            label_to_idx=label_to_idx,
            img_size=int(timm_cfg.get("img_size", 224)),
            train=split_name == "train",
            use_mask_crop=bool(timm_cfg.get("use_mask_crop", True)),
            mask_threshold=int(timm_cfg.get("mask_threshold", 127)),
            mask_padding=int(timm_cfg.get("mask_padding", 4)),
            normalize=bool(timm_cfg.get("normalize", True)),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(timm_cfg.get("batch_size", 32)),
            shuffle=split_name == "train",
            num_workers=int(timm_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return loaders, class_names, label_to_idx, merged


def prepare_single_tensor(image_path: str, mask_path: str | None, config, device=None):
    timm_cfg = config["timm"]
    # 用和测试时评估一样的预处理，确保推理和基准测试保持一致。
    image = _prepare_pil_image(
        image_path,
        mask_path,
        bool(timm_cfg.get("use_mask_crop", True)),
        int(timm_cfg.get("mask_threshold", 127)),
        int(timm_cfg.get("mask_padding", 4)),
    )
    transform = build_transforms(
        int(timm_cfg.get("img_size", 224)),
        train=False,
        normalize=bool(timm_cfg.get("normalize", True)),
    )
    tensor = transform(image).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
