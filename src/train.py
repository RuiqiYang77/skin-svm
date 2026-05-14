# 训练示例：
# python src/train.py --config config/svm.yaml --experiment_id svm_exp001
#
# 参数说明：
# --config: YAML 配置文件路径
# --experiment_id: 实验编号，所有输出会保存到 outputs/{experiment_id}/

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.dataset import build_metadata, validate_metadata
from src.dataloader.features import extract_feature_table
from src.dataloader.split import create_grouped_split, split_features
from src.model.svm import save_model_bundle, train_svm
from src.utils.config import load_config, save_config
from src.utils.evaluation import (
    augmentation_robustness,
    build_predictions_frame,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.utils.io import ensure_dir, save_json


def _print_split_summary(metrics, labels):
    print("Split summary (accuracy / balanced_acc / macro_f1):")
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics[split_name]
        print(
            f"  {split_name:5s}  "
            f"acc={split_metrics['accuracy']:.4f}  "
            f"bal_acc={split_metrics['balanced_accuracy']:.4f}  "
            f"macro_f1={split_metrics['macro_f1']:.4f}"
        )

    test_report = metrics["test"]["classification_report"]
    recall_parts = []
    for label in labels:
        label_report = test_report.get(label, {})
        if "recall" in label_report:
            recall_parts.append(f"{label}={label_report['recall']:.4f}")
    if recall_parts:
        print("Test class recall: " + ", ".join(recall_parts))

    robustness = metrics["test"].get("augmentation_robustness", {})
    if robustness:
        consistency = robustness.get("prediction_consistency")
        if consistency is not None:
            print(f"Test augmentation consistency: {consistency:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SVM lesion classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--experiment_id",
        required=True,
        help="Experiment id. Outputs are saved to outputs/{experiment_id}/.",
    )
    parser.add_argument(
        "--reuse_features",
        action="store_true",
        help="Reuse outputs/{experiment_id}/features.csv when it exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("model") != "svm":
        raise ValueError("This training entry currently supports model: svm only.")

    output_dir = ensure_dir(Path(config["data"]["output_dir"]) / args.experiment_id)
    save_config(config, output_dir / "config.yaml")

    metadata = build_metadata(config)
    validate_metadata(metadata)

    features_path = output_dir / "features.csv"
    if args.reuse_features and features_path.exists():
        features_df = pd.read_csv(features_path)
    else:
        features_df = extract_feature_table(metadata, config)
        features_df.to_csv(features_path, index=False)

    split_df = create_grouped_split(metadata, config)
    split_df.to_csv(output_dir / "split.csv", index=False)

    split_data = split_features(features_df, split_df)
    X_train, y_train, train_meta = split_data["train"]
    X_val, y_val, val_meta = split_data["val"]
    X_test, y_test, test_meta = split_data["test"]

    feature_columns = X_train.columns.tolist()
    model = train_svm(
        X_train,
        y_train,
        groups=train_meta["base_id"].to_numpy(),
        config=config,
    )

    labels = sorted(metadata["label"].unique().tolist())
    metrics = {}
    prediction_frames = []

    for split_name, X, y, meta in [
        ("train", X_train, y_train, train_meta),
        ("val", X_val, y_val, val_meta),
        ("test", X_test, y_test, test_meta),
    ]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        metrics[split_name] = evaluate_predictions(y, y_pred, labels)
        pred_frame = build_predictions_frame(meta, y_pred, y_prob, model.classes_)
        pred_frame["split"] = split_name
        prediction_frames.append(pred_frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    test_predictions = predictions[predictions["split"] == "test"].copy()
    robustness = augmentation_robustness(test_predictions)
    robustness_detail = robustness.pop("detail")
    robustness_detail.to_csv(output_dir / "robustness_detail.csv", index=False)
    metrics["test"]["augmentation_robustness"] = robustness

    save_json(metrics, output_dir / "metrics.json")
    save_confusion_matrix(
        test_predictions["label"],
        test_predictions["pred_label"],
        labels,
        output_dir / "confusion_matrix.png",
    )
    save_model_bundle(model, feature_columns, output_dir / "model.joblib")

    print(f"Experiment finished: {output_dir}")
    _print_split_summary(metrics, labels)


if __name__ == "__main__":
    main()
