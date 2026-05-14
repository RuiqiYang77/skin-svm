"""timm 对比基线的训练和评估工具。

用途：
    把微调循环、指标计算、checkpoint 保存、split 评估都集中在一处，
    这样单模型入口和批量脚本都能复用同一套逻辑。

主要入口：
    set_seed(...)
    train_model(...)
    evaluate_model(...)
    compute_metrics(...)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "dip_project_timm_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.timm_baseline.model import save_checkpoint


def set_seed(seed: int):
    # 尽可能让基线在多次运行间可复现（看后端支持程度）。
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(labels, preds, probs, num_classes: int, class_names):
    # 这些是和前面实验脚本用过的对比指标完全一样。
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    report = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    specificities = []
    for idx in range(num_classes):
        tn = cm.sum() - (cm[idx, :].sum() + cm[:, idx].sum() - cm[idx, idx])
        fp = cm[:, idx].sum() - cm[idx, idx]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    auc_macro = None
    per_class_auc = []
    if probs is not None and len(probs) > 0:
        try:
            auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc_macro = None
        for idx in range(num_classes):
            try:
                auc_i = roc_auc_score((labels == idx).astype(int), probs[:, idx])
            except Exception:
                auc_i = None
            per_class_auc.append(auc_i)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "specificity": float(np.mean(specificities)) if specificities else 0.0,
        "kappa": float(cohen_kappa_score(labels, preds)),
        "auc_macro": None if auc_macro is None else float(auc_macro),
        "per_class_auc": per_class_auc,
    }


def save_confusion_matrix(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds, labels=class_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_classification_report_text(labels, preds, class_names, output_path, title=None):
    text = classification_report(
        labels,
        preds,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"{title}\n")
            f.write("=" * max(30, len(title)) + "\n")
        f.write(text)


def _run_epoch(model, loader, device, criterion=None, optimizer=None):
    # 训练和评估共用的循环；optimizer=None 表示推理模式。
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    preds_all = []
    probs_all = []
    records = []

    for images, labels, meta in loader:
        images = images.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels) if criterion is not None else None
            if training:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.tolist())
        probs_all.append(probs)
        records.append(pd.DataFrame(meta))

        if loss is not None:
            total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)

    probs_all = np.concatenate(probs_all, axis=0) if probs_all else None
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, np.array(labels_all), np.array(preds_all), probs_all, records_df


def train_model(
    model,
    loaders,
    device,
    config,
    class_names,
    label_to_idx,
    architecture,
    checkpoint_path,
):
    timm_cfg = config["timm"]
    # 标准微调配方：Adam + 交叉熵 + ReduceLROnPlateau + 早停。
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(timm_cfg.get("label_smoothing", 0.0))
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(timm_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(timm_cfg.get("weight_decay", 0.0)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(timm_cfg.get("lr_factor", 0.5)),
        patience=int(timm_cfg.get("lr_patience", 3)),
    )

    epochs = int(timm_cfg.get("num_epochs", 50))
    patience = int(timm_cfg.get("patience", 7))
    monitor_metric = str(timm_cfg.get("monitor_metric", "macro_f1"))

    history = []
    best_score = -float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_labels, train_preds, train_probs, _ = _run_epoch(
            model,
            loaders["train"],
            device,
            criterion=criterion,
            optimizer=optimizer,
        )
        val_loss, val_labels, val_preds, val_probs, _ = _run_epoch(
            model,
            loaders["val"],
            device,
            criterion=criterion,
            optimizer=None,
        )

        train_metrics = compute_metrics(
            train_labels, train_preds, train_probs, len(class_names), class_names
        )
        val_metrics = compute_metrics(
            val_labels, val_preds, val_probs, len(class_names), class_names
        )

        current_score = float(val_metrics.get(monitor_metric, val_metrics["macro_f1"]))
        scheduler.step(current_score)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if current_score > best_score:
            # 立即保存最佳 checkpoint，这样预测脚本后面可以直接复用。
            best_score = current_score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                checkpoint_path,
                model,
                class_names,
                label_to_idx,
                architecture,
                config,
                best_score,
                best_epoch,
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered, stopping training.")
            break

    return pd.DataFrame(history), {"best_score": best_score, "best_epoch": best_epoch}


def evaluate_model(model, loader, device, class_names):
    # 既返回标量指标，也返回逐行预测表供 CSV 导出。
    loss, labels, preds, probs, records = _run_epoch(
        model,
        loader,
        device,
        criterion=None,
        optimizer=None,
    )
    metrics = compute_metrics(labels, preds, probs, len(class_names), class_names)
    predictions = records.copy()
    predictions["pred_label"] = [class_names[idx] for idx in preds]
    if probs is not None:
        for idx, label in enumerate(class_names):
            predictions[f"prob_{label}"] = probs[:, idx]
    return metrics, predictions, labels, preds, probs
