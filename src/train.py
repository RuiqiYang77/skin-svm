"""Train a configured skin lesion classifier and export experiment artifacts.

Example:
  python src/train.py --config config/svm.yaml --experiment_id svm_exp001
"""

import argparse
import copy
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.dataset import build_metadata, validate_metadata
from src.dataloader.split import create_grouped_split, split_features
from src.utils.config import load_config, save_config
from src.utils.evaluation import (
    augmentation_robustness,
    build_predictions_frame,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.utils.io import ensure_dir, save_json


def parse_args():
    """Parse training options shared by all supported model backends."""
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
    parser.add_argument(
        "--features_module",
        default="src.dataloader.features",
        help="Python module path for feature extraction (e.g. src.dataloader.features_xlc).",
    )
    return parser.parse_args()


def main():
    """Run the full experiment: metadata, features, split, training, and export."""
    args = parse_args()
    config = load_config(args.config)

    model_name = config.get("model")
    if model_name is None:
        raise ValueError("Config must specify a 'model' field.")
    model_module = importlib.import_module(f"src.model.{model_name}")
    train_func = getattr(model_module, f"train_{model_name}")
    save_model_bundle = getattr(model_module, "save_model_bundle")

    output_dir = ensure_dir(Path(config["data"]["output_dir"]) / args.experiment_id)

    # Feature extraction is imported dynamically so experiments can swap modules.
    features_mod = importlib.import_module(args.features_module)
    extract_feature_table = features_mod.extract_feature_table

    metadata = build_metadata(config)
    has_aug = (metadata["augmentation_id"] != "original").any()
    validate_metadata(metadata, strict_groups=has_aug)

    features_path = output_dir / "features.csv"
    if args.reuse_features and features_path.exists():
        features_df = pd.read_csv(features_path)
    else:
        # Cache tabular features because extraction is the slowest deterministic step.
        features_df = extract_feature_table(metadata, config)
        features_df.to_csv(features_path, index=False)

    labels = sorted(metadata["label"].unique().tolist())
    split_search_cfg = config[model_name].get("split_search", {})

    def run_once(config_run):
        """Train and evaluate one split configuration."""
        split_df = create_grouped_split(metadata, config_run)
        split_data = split_features(features_df, split_df)
        X_train, y_train, train_meta = split_data["train"]
        X_val, y_val, val_meta = split_data["val"]
        X_test, y_test, test_meta = split_data["test"]

        feature_columns = X_train.columns.tolist()
        model = train_func(
            X_train, y_train,
            groups=train_meta["base_id"].to_numpy(),
            config=config_run,
            X_val=X_val, y_val=y_val,
        )

        metrics = {}
        prediction_frames = []
        # Evaluate each split with identical metrics for paper-ready reporting.
        for split_name, X, y, meta in [
            ("train", X_train, y_train, train_meta),
            ("val", X_val, y_val, val_meta),
            ("test", X_test, y_test, test_meta),
        ]:
            if len(X) == 0:
                metrics[split_name] = {
                    "accuracy": None, "balanced_accuracy": None,
                    "macro_precision": None, "macro_recall": None,
                    "macro_f1": None, "classification_report": {},
                }
                continue
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            metrics[split_name] = evaluate_predictions(y, y_pred, labels)
            pred_frame = build_predictions_frame(meta, y_pred, y_prob, model.classes_)
            pred_frame["split"] = split_name
            prediction_frames.append(pred_frame)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        test_predictions = predictions[predictions["split"] == "test"].copy()
        if not test_predictions.empty:
            # Measure whether predictions are stable under image augmentation.
            robustness = augmentation_robustness(test_predictions)
            robustness_detail = robustness.pop("detail")
            metrics["test"]["augmentation_robustness"] = robustness
        else:
            robustness_detail = pd.DataFrame()

        return {
            "split_df": split_df, "feature_columns": feature_columns,
            "model": model, "metrics": metrics,
            "predictions": predictions, "test_predictions": test_predictions,
            "robustness_detail": robustness_detail,
        }

    if split_search_cfg.get("enabled", False):
        candidates = split_search_cfg.get("candidates", [])
        if not candidates:
            raise ValueError("split_search.enabled is true but no candidates provided.")

        metric_split = split_search_cfg.get("metric_split", "val")
        metric_name = split_search_cfg.get("metric", "macro_f1")
        summary_rows = []
        best_result = None

        for idx, candidate in enumerate(candidates, start=1):
            config_run = copy.deepcopy(config)
            config_run[model_name]["split"].update(candidate)
            result = run_once(config_run)

            if metric_split not in result["metrics"]:
                raise ValueError(f"Unknown metric split: {metric_split}")
            if metric_name not in result["metrics"][metric_split]:
                raise ValueError(f"Unknown metric name: {metric_name}")

            score = result["metrics"][metric_split][metric_name]
            split_cfg = config_run[model_name]["split"]
            summary_rows.append({
                "candidate": idx,
                "train_size": split_cfg.get("train_size"),
                "val_size": split_cfg.get("val_size"),
                "test_size": split_cfg.get("test_size"),
                "val_macro_f1": result["metrics"]["val"]["macro_f1"],
                "test_macro_f1": result["metrics"]["test"]["macro_f1"],
                "test_balanced_accuracy": result["metrics"]["test"]["balanced_accuracy"],
                f"{metric_split}_{metric_name}": score,
            })

            if best_result is None or score > best_result["score"]:
                best_result = {
                    "score": score, "config": config_run,
                    "result": result, "candidate": candidate,
                }

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "split_search_summary.csv", index=False)
        print(f"Split search summary:\n{summary_df.to_string(index=False)}")

        final = best_result["result"]
        best_candidate = best_result["candidate"]
        config = best_result["config"]
        save_config(config, output_dir / "config.yaml")
        if best_candidate.get("val_size") is not None:
            print(f"Best split: train={best_candidate['train_size']}, "
                  f"val={best_candidate['val_size']}, test={best_candidate['test_size']}")
    else:
        final = run_once(config)
        save_config(config, output_dir / "config.yaml")

    final["predictions"].to_csv(output_dir / "predictions.csv", index=False)
    final["split_df"].to_csv(output_dir / "split.csv", index=False)
    final["robustness_detail"].to_csv(output_dir / "robustness_detail.csv", index=False)
    save_json(final["metrics"], output_dir / "metrics.json")
    save_model_bundle(final["model"], final["feature_columns"], output_dir / "model.joblib")

    has_test = not final["test_predictions"].empty
    if has_test:
        save_confusion_matrix(
            final["test_predictions"]["label"],
            final["test_predictions"]["pred_label"],
            labels,
            output_dir / "confusion_matrix.png",
        )

    print(f"Experiment finished: {output_dir}")
    if has_test:
        print(f"Test macro F1: {final['metrics']['test']['macro_f1']:.4f}")
        print(f"Test balanced accuracy: {final['metrics']['test']['balanced_accuracy']:.4f}")
    else:
        print(f"Val macro F1: {final['metrics']['val']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
