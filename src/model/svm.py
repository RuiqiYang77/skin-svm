"""Define the SVM training backend used by the main experiment pipeline.

The backend builds a scikit-learn pipeline, performs optional grouped grid
search, and saves the trained model with its feature-column schema.
"""

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(config):
    """Build the SVM pipeline used by the main experiments."""
    svm_cfg = config["svm"]
    steps = [("scaler", StandardScaler())]

    if svm_cfg["pca"].get("enabled", False):
        steps.append(
            (
                "pca",
                PCA(n_components=svm_cfg["pca"].get("n_components", 0.95)),
            )
        )

    model_cfg = svm_cfg["model"]
    steps.append(
        (
            "svc",
            SVC(
                kernel=model_cfg.get("kernel", "rbf"),
                class_weight=model_cfg.get("class_weight", "balanced"),
                probability=model_cfg.get("probability", True),
                random_state=model_cfg.get("random_state", 42),
            ),
        )
    )
    return Pipeline(steps)


def train_svm(X_train, y_train, groups, config, X_val=None, y_val=None):
    """Train an SVM with grouped CV and optional validation-set selection."""
    pipeline = build_svm_pipeline(config)
    grid_cfg = config["svm"]["grid_search"]

    if not grid_cfg.get("enabled", True):
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = {
        "svc__C": grid_cfg.get("C", [1]),
        "svc__gamma": grid_cfg.get("gamma", ["scale"]),
    }
    cv = StratifiedGroupKFold(
        n_splits=grid_cfg.get("cv", 5),
        shuffle=True,
        random_state=config["svm"]["split"].get("random_state", 42),
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=grid_cfg.get("scoring", "f1_macro"),
        cv=cv,
        n_jobs=grid_cfg.get("n_jobs", 1),
        refit=False,  # we will pick best model via validation set
    )
    search.fit(X_train, y_train, groups=groups)

    # Model selection: use validation set if provided, else fall back to CV
    if X_val is not None and y_val is not None and grid_cfg.get("val_selection", True):
        import numpy as np
        from sklearn.metrics import f1_score

        best_score = -1.0
        best_idx = 0
        for i, params in enumerate(search.cv_results_["params"]):
            # Fit a fresh pipeline with these params on full training data
            pipe = build_svm_pipeline(config)
            pipe.set_params(**params)
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            if val_f1 > best_score:
                best_score = val_f1
                best_idx = i

        best_params = search.cv_results_["params"][best_idx]
        print(f"  CV best: {search.best_params_} (CV={search.best_score_:.4f})")
        print(f"  Val best: {best_params} (Val F1={best_score:.4f})")

        # Refit with validation-best params on train+val combined
        pipeline.set_params(**best_params)
        pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return pipeline
    else:
        # Fall back: refit with CV-best params
        best_params = search.best_params_
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        return pipeline


def save_model_bundle(model, feature_columns, path):
    """Persist the trained model and feature column order."""
    joblib.dump({"model": model, "feature_columns": feature_columns}, path)


def load_model_bundle(path):
    """Load a model bundle saved by ``save_model_bundle``."""
    return joblib.load(path)
