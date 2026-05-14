"""对多个 timm 模型运行病变分类对比基线。

用途：
    在几个不同的 timm 骨干网上执行同一套训练流程，
    最后生成一份紧凑的对比总表。

使用方法：
    python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id exp001

可选用法（只跑部分模型）：
    python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id exp001 \
        --models efficientnet_b0 resnet18

输出文件：
    outputs/timm_{experiment_id}/
        suite_summary.csv
        <model_name>/...
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_timm import run_experiment
from src.utils.config import load_config


DEFAULT_MODELS = [
    "efficientnet_b0",
    "resnet18",
    "vit_base_patch16_224",
    "samvit_base_patch16",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a timm baseline suite across multiple models.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--experiment_id",
        required=True,
        help="Suite id. Results will be saved under outputs/timm_{experiment_id}/.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Optional subset of model names to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # 共享的批量根目录：每个模型都在下面建独立子目录。
    suite_root = Path(config["data"]["output_dir"]) / f"timm_{args.experiment_id}"
    suite_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in args.models:
        # 复用同一套训练流程，只需要替换模型名。
        print("=" * 80)
        print(f"Running timm baseline model: {model_name}")
        model_config = {**config, "timm": dict(config["timm"])}
        model_config["timm"]["model_name"] = model_name
        output_dir, metrics = run_experiment(
            model_config,
            args.experiment_id,
            model_name_override=model_name,
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "output_dir": str(output_dir),
                "test_accuracy": metrics["test"]["accuracy"],
                "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
            }
        )

    import pandas as pd

    # 一份紧凑的 CSV 就足够做模型对比了。
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(suite_root / "suite_summary.csv", index=False)
    print("=" * 80)
    print(f"Suite finished. Summary saved to: {suite_root / 'suite_summary.csv'}\n")
    
    # 只显示对比指标，隐藏冗长的 output_dir 列，便于对齐。
    display_df = summary_df[["model_name", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"]].copy()
    display_df.columns = ["模型名称", "测试准确率", "测试平衡准确率", "宏平均 F1"]
    display_df["测试准确率"] = display_df["测试准确率"].apply(lambda x: f"{x:.6f}")
    display_df["测试平衡准确率"] = display_df["测试平衡准确率"].apply(lambda x: f"{x:.6f}")
    display_df["宏平均 F1"] = display_df["宏平均 F1"].apply(lambda x: f"{x:.6f}")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()