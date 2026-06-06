import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(config):
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


def train_svm(X_train, y_train, groups, config):
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
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_


def save_model_bundle(model, feature_columns, path):
    joblib.dump({"model": model, "feature_columns": feature_columns}, path)


def load_model_bundle(path):
    return joblib.load(path)
