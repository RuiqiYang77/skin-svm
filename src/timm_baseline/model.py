"""timm 对比基线的模型创建和 checkpoint 工具。

用途：
    把 timm 模型构造和 checkpoint 序列化集中在一处，
    这样训练和推理都能总是用同一个骨干和类别映射。

主要入口：
    create_model(...)
    save_checkpoint(...)
    load_model_from_checkpoint(...)
"""

from __future__ import annotations

from pathlib import Path

import torch
import timm


def create_model(architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    # timm 模型创建集中在一处，这样不同脚本都用同一套架构设置。
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if dropout is not None:
        kwargs["drop_rate"] = dropout
    try:
        return timm.create_model(architecture, **kwargs)
    except TypeError:
        kwargs.pop("drop_rate", None)
        return timm.create_model(architecture, **kwargs)


def save_checkpoint(
    path,
    model,
    class_names,
    label_to_idx,
    architecture,
    config,
    best_metric,
    epoch,
):
    # 存足够的元数据，这样后面能精确恢复对比模型，不用猜架构。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": architecture,
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "model_state_dict": model.state_dict(),
        "best_metric": float(best_metric),
        "epoch": int(epoch),
        "timm_config": config["timm"],
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)


def load_model_from_checkpoint(path, device=None):
    # 恢复训练时用的精确模型定义。
    checkpoint = load_checkpoint(path, map_location=device)
    class_names = checkpoint["class_names"]
    architecture = checkpoint["architecture"]
    timm_cfg = checkpoint.get("timm_config", {})
    model = create_model(
        architecture,
        num_classes=len(class_names),
        pretrained=False,
        dropout=float(timm_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, checkpoint
