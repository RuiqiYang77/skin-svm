"""用 timm 对比基线做单张图像推理。

用途：
    加载训练好的 timm checkpoint，对单张病灶图像做推理，
    用的预处理和测试集评估时完全一致。

使用方法：
    python src/predict_timm.py --config config/timm_baseline.yaml --experiment_id exp001 \
        --image_path data/image/1.jpg --mask_path data/mask/mask_1.jpg

输出文件：
    outputs/timm_{experiment_id}/single_prediction.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.timm_baseline.data import prepare_single_tensor
from src.timm_baseline.model import load_model_from_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a single lesion image with a timm baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--experiment_id",
        required=True,
        help="Experiment id. Loads outputs/timm_{experiment_id}/model_best.pth.",
    )
    parser.add_argument("--image_path", required=True, help="Input image path.")
    parser.add_argument("--mask_path", required=True, help="Input mask path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(Path(config["data"]["output_dir"]) / f"timm_{args.experiment_id}")
    checkpoint_path = output_dir / "model_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = config["timm"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    class_names = checkpoint["class_names"]

    # 预测路径镜像了测试时的预处理和 mask 裁剪。
    image_tensor = prepare_single_tensor(args.image_path, args.mask_path, config, device=device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(probabilities.argmax())
        pred_label = class_names[pred_idx]

    result = {
        "image_path": args.image_path,
        "mask_path": args.mask_path,
        "pred_label": pred_label,
    }
    for idx, label in enumerate(class_names):
        result[f"prob_{label}"] = float(probabilities[idx])

    prediction_path = output_dir / "single_prediction.csv"
    pd.DataFrame([result]).to_csv(prediction_path, index=False)
    print(result)
    print(f"Saved prediction to: {prediction_path}")


if __name__ == "__main__":
    main()
