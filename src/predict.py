# 预测示例：
# python src/predict.py \
#   --config config/svm.yaml \
#   --experiment_id svm_exp001 \
#   --image_path data/image/1.jpg \
#   --mask_path data/mask/mask_1.jpg
#
# 参数说明：
# --config: YAML 配置文件路径
# --experiment_id: 要使用的实验编号，会读取 outputs/{experiment_id}/model.joblib
# --image_path: 待预测图像路径
# --mask_path: 待预测 mask 路径

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.svm import load_model_bundle
from src.utils.config import load_config
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one lesion image.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True, help="Experiment id.")
    parser.add_argument("--image_path", required=True, help="Input image path.")
    parser.add_argument("--mask_path", required=True, help="Input mask path.")
    parser.add_argument(
        "--features_module",
        default="src.dataloader.features",
        help="Python module path for feature extraction (must match the trained model).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(Path(config["data"]["output_dir"]) / args.experiment_id)
    bundle = load_model_bundle(output_dir / "model.joblib")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    features_mod = importlib.import_module(args.features_module)
    extract_features = features_mod.extract_features

    feature_dict = extract_features(args.image_path, args.mask_path, config)
    X = pd.DataFrame([feature_dict]).reindex(columns=feature_columns, fill_value=0.0)

    pred_label = model.predict(X)[0]
    result = {
        "image_path": args.image_path,
        "mask_path": args.mask_path,
        "pred_label": pred_label,
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        for label, prob in zip(model.classes_, probs):
            result[f"prob_{label}"] = float(prob)

    prediction_path = output_dir / "single_prediction.csv"
    pd.DataFrame([result]).to_csv(prediction_path, index=False)
    print(result)
    print(f"Saved prediction to: {prediction_path}")


if __name__ == "__main__":
    main()
