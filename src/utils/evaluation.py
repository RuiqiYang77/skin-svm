"""Compute metrics and save evaluation artifacts for lesion classification.

The helpers produce scalar classification metrics, prediction tables,
confusion-matrix figures, and augmentation-consistency summaries.
"""

import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "dip_project_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(y_true, y_pred, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
    }


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_predictions_frame(metadata_df, y_pred, y_prob=None, class_labels=None):
    predictions = metadata_df.copy()
    predictions["pred_label"] = y_pred
    if y_prob is not None and class_labels is not None:
        for idx, label in enumerate(class_labels):
            predictions[f"prob_{label}"] = y_prob[:, idx]
    return predictions


def augmentation_robustness(predictions_df):
    rows = []
    for base_id, group in predictions_df.groupby("base_id"):
        original = group[group["augmentation_id"] == "original"]
        augmented = group[group["augmentation_id"] != "original"]
        if original.empty or augmented.empty:
            continue

        original_pred = original.iloc[0]["pred_label"]
        augmented_preds = augmented["pred_label"].tolist()
        all_preds = [original_pred] + augmented_preds
        rows.append(
            {
                "base_id": base_id,
                "label": original.iloc[0]["label"],
                "original_pred": original_pred,
                "augmented_preds": "|".join(augmented_preds),
                "is_consistent": len(set(all_preds)) == 1,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "prediction_consistency": None,
            "num_groups": 0,
            "detail": detail,
        }

    original_df = predictions_df[predictions_df["augmentation_id"] == "original"]
    augmented_df = predictions_df[predictions_df["augmentation_id"] != "original"]

    return {
        "prediction_consistency": float(detail["is_consistent"].mean()),
        "num_groups": int(len(detail)),
        "original_accuracy": float(
            accuracy_score(original_df["label"], original_df["pred_label"])
        ),
        "augmented_accuracy": float(
            accuracy_score(augmented_df["label"], augmented_df["pred_label"])
        ),
        "detail": detail,
    }
