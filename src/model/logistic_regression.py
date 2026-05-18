import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_lr_pipeline(config):
    """Build a Logistic Regression pipeline with optional scaling."""
    lr_cfg = config["logistic_regression"]
    steps = [("scaler", StandardScaler())]

    model_cfg = lr_cfg["model"]
    steps.append(
        (
            "lr",
            LogisticRegression(
                penalty=model_cfg.get("penalty", "l2"),
                C=model_cfg.get("C", 1.0),
                class_weight=model_cfg.get("class_weight", "balanced"),
                random_state=model_cfg.get("random_state", 42),
                max_iter=model_cfg.get("max_iter", 1000),
                solver=model_cfg.get("solver", "lbfgs"),
                n_jobs=model_cfg.get("n_jobs", 1),
            ),
        )
    )
    return Pipeline(steps)


def train_logistic_regression(X_train, y_train, groups, config):
    """Train a Logistic Regression model with optional grid search.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    groups : np.ndarray
        Group labels (not used by LogisticRegression, kept for interface consistency).
    config : dict
        Full configuration dictionary.

    Returns
    -------
    Pipeline
        Trained pipeline (best estimator if grid search is enabled).
    """
    pipeline = build_lr_pipeline(config)
    lr_cfg = config["logistic_regression"]
    grid_cfg = lr_cfg.get("grid_search", {})

    if not grid_cfg.get("enabled", True):
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = {
        "lr__C": grid_cfg.get("C", [0.01, 0.1, 1, 10, 100]),
    }

    cv = StratifiedKFold(
        n_splits=grid_cfg.get("cv", 5),
        shuffle=True,
        random_state=lr_cfg["split"].get("random_state", 42),
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
