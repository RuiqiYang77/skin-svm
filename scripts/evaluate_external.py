"""
对独立测试集进行批量评估。

用法：
  python scripts/evaluate_external.py \
      --config config/svm_external.yaml \
      --experiment_id svm_external_001 \
      --test_csv data/external/test/label.csv \
      --test_image_dir data/external/test/image \
      --test_mask_dir data/external/test/mask
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.features import extract_features
from src.model.svm import load_model_bundle
from src.utils.config import load_config
from src.utils.evaluation import (
    evaluate_predictions,
    save_confusion_matrix,
)
from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on external test set.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--experiment_id",
        required=True,
        help="Experiment id (reads outputs/{experiment_id}/model.joblib).",
    )
    parser.add_argument("--test_csv", required=True, help="Test set label.csv path.")
    parser.add_argument(
        "--test_image_dir", required=True, help="Test set image directory."
    )
    parser.add_argument(
        "--test_mask_dir", required=True, help="Test set mask directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = ensure_dir(Path(config["data"]["output_dir"]) / args.experiment_id)

    # 加载训练好的模型
    bundle = load_model_bundle(output_dir / "model.joblib")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    print(f"Loaded model from {output_dir / 'model.joblib'}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}...")

    # 读取测试集标签
    test_df = pd.read_csv(args.test_csv)
    test_image_dir = Path(args.test_image_dir)
    test_mask_dir = Path(args.test_mask_dir)

    print(f"\nTest set: {len(test_df)} images")
    print(test_df["dx"].value_counts().to_string())

    # 逐张提取特征并预测
    y_true = []
    y_pred = []
    y_prob_list = []
    results = []

    for row in test_df.itertuples(index=False):
        image_id = str(row.image_id)
        image_path = str(test_image_dir / f"{image_id}.jpg")
        mask_path = str(test_mask_dir / f"mask_{image_id}.jpg")

        # 提取特征
        feature_dict = extract_features(image_path, mask_path, config)
        X = (
            pd.DataFrame([feature_dict])
            .reindex(columns=feature_columns, fill_value=0.0)
        )

        # 预测
        pred = model.predict(X)[0]
        y_true.append(row.dx)
        y_pred.append(pred)

        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0]
            y_prob_list.append(prob)

        result = {
            "image_id": image_id,
            "true_label": row.dx,
            "pred_label": pred,
        }
        if prob is not None:
            for label, p in zip(model.classes_, prob):
                result[f"prob_{label}"] = float(p)
        results.append(result)

    # 保存预测详情
    pred_df = pd.DataFrame(results)
    pred_csv = output_dir / "external_test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nPredictions saved: {pred_csv}")

    # 计算指标
    labels = sorted(test_df["dx"].unique().tolist())
    metrics = evaluate_predictions(y_true, y_pred, labels)
    print(f"\n{'='*50}")
    print("外部测试集评估结果")
    print(f"{'='*50}")
    for key, val in metrics.items():
        if key != "classification_report":
            print(f"  {key}: {val:.4f}")
    print(f"\n分类报告:")
    for cls, report in metrics["classification_report"].items():
        if isinstance(report, dict):
            print(
                f"  {cls:6s}: precision={report['precision']:.4f}, "
                f"recall={report['recall']:.4f}, "
                f"f1={report['f1-score']:.4f}, "
                f"support={int(report['support'])}"
            )

    # 保存指标
    metrics_json = output_dir / "external_test_metrics.json"
    save_json(metrics, metrics_json)
    print(f"\nMetrics saved: {metrics_json}")

    # 混淆矩阵
    cm_path = output_dir / "external_test_confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, labels, cm_path)
    print(f"Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    main()
