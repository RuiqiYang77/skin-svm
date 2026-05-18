import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_rf_pipeline(config):
    """Build a Random Forest pipeline (scaler + classifier)."""
    rf_cfg = config["random_forest"]
    steps = [("scaler", StandardScaler())]

    model_cfg = rf_cfg["model"]
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=model_cfg.get("n_estimators", 100),
                max_depth=model_cfg.get("max_depth", None),
                min_samples_split=model_cfg.get("min_samples_split", 2),
                min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
                class_weight=model_cfg.get("class_weight", "balanced"),
                random_state=model_cfg.get("random_state", 42),
                n_jobs=model_cfg.get("n_jobs", -1),
            ),
        )
    )
    return Pipeline(steps)


def train_random_forest(X_train, y_train, groups, config):
    """Train a Random Forest model with optional grid search.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    groups : np.ndarray
        Group labels (not used by RandomForest, kept for interface consistency).
    config : dict
        Full configuration dictionary.

    Returns
    -------
    Pipeline
        Trained pipeline (best estimator if grid search is enabled).
    """
    pipeline = build_rf_pipeline(config)
    rf_cfg = config["random_forest"]
    grid_cfg = rf_cfg.get("grid_search", {})

    if not grid_cfg.get("enabled", True):
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = {
        "rf__n_estimators": grid_cfg.get("n_estimators", [50, 100, 200]),
        "rf__max_depth": grid_cfg.get("max_depth", [None, 10, 20, 30]),
    }

    cv = StratifiedKFold(
        n_splits=grid_cfg.get("cv", 5),
        shuffle=True,
        random_state=rf_cfg["split"].get("random_state", 42),
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=grid_cfg.get("scoring", "f1_macro"),
        cv=cv,
        n_jobs=grid_cfg.get("n_jobs", 1),
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def save_model_bundle(model, feature_columns, path):
    """Save trained model and feature column names to disk."""
    joblib.dump({"model": model, "feature_columns": feature_columns}, path)


def load_model_bundle(path):
    """Load a saved model bundle from disk."""
    return joblib.load(path)
