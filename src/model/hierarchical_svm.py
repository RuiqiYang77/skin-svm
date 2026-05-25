"""
Hierarchical SVM classifier.

Two-stage pipeline:
  Stage 1 (SVM1): vasc  vs  (nv + mel)   — easy separation, near-perfect
  Stage 2 (SVM2): nv    vs  mel           — hard pair, dedicated classifier

Prediction flow:
  SVM1 predicts vasc → return "vasc"
  SVM1 predicts non-vasc → route to SVM2 → return "nv" or "mel"
"""

import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Scorer resolver
# ---------------------------------------------------------------------------

def _resolve_scorer(scoring):
    """
    Resolve scorer name to a callable if it's a binary f1 shorthand.
    e.g. "f1_vasc" → make_scorer(f1_score, pos_label="vasc", average="binary")
         "f1_mel"  → make_scorer(f1_score, pos_label="mel",  average="binary")
    Standard sklearn strings (e.g. "f1_macro") are passed through as-is.
    """
    if isinstance(scoring, str) and scoring.startswith("f1_"):
        suffix = scoring[3:]
        if suffix not in ("macro", "micro", "weighted", "binary", "samples"):
            return make_scorer(f1_score, pos_label=suffix, average="binary")
    return scoring


# ---------------------------------------------------------------------------
# Pipeline builder (shared by both stages)
# ---------------------------------------------------------------------------

def _build_pipeline(stage_cfg):
    steps = [("scaler", StandardScaler())]

    fs_cfg = stage_cfg.get("feature_selection", {})
    if fs_cfg.get("enabled", False):
        steps.append((
            "selector",
            SelectKBest(mutual_info_classif, k=fs_cfg.get("k", 100)),
        ))

    pca_cfg = stage_cfg.get("pca", {})
    if pca_cfg.get("enabled", False):
        steps.append(("pca", PCA(n_components=pca_cfg.get("n_components", 0.95))))

    model_cfg = stage_cfg.get("model", {})
    steps.append((
        "svc",
        SVC(
            kernel=model_cfg.get("kernel", "rbf"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            probability=model_cfg.get("probability", True),
            random_state=model_cfg.get("random_state", 42),
        ),
    ))
    return Pipeline(steps)


def _train_stage(X, y, groups, stage_cfg, random_state=42):
    pipeline = _build_pipeline(stage_cfg)
    grid_cfg = stage_cfg.get("grid_search", {})

    if not grid_cfg.get("enabled", True):
        pipeline.fit(X, y)
        return pipeline

    param_grid = {
        "svc__C": grid_cfg.get("C", [1]),
        "svc__gamma": grid_cfg.get("gamma", ["scale"]),
    }
    cv = StratifiedGroupKFold(
        n_splits=grid_cfg.get("cv", 5),
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=_resolve_scorer(grid_cfg.get("scoring", "f1_macro")),
        cv=cv,
        n_jobs=grid_cfg.get("n_jobs", 1),
        refit=True,
    )
    search.fit(X, y, groups=groups)
    print(f"  Best params: {search.best_params_}  CV score: {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# HierarchicalSVM
# ---------------------------------------------------------------------------

class HierarchicalSVM:
    """
    Two-stage hierarchical SVM.

    Attributes
    ----------
    vasc_label : str
        The label that SVM1 separates from the rest (default "vasc").
    svm1 : fitted Pipeline
        Binary classifier: vasc vs rest.
    svm2 : fitted Pipeline
        Binary classifier: nv vs mel (trained only on non-vasc samples).
    classes_ : list
        All class labels in sorted order.
    """

    def __init__(self, vasc_label="vasc", vasc_threshold=0.5):
        self.vasc_label = vasc_label
        self.vasc_threshold = vasc_threshold
        self.svm1 = None
        self.svm2 = None
        self.classes_ = None

    def fit(self, X_train, y_train, groups, config):
        random_state = config["hierarchical_svm"]["split"].get("random_state", 42)
        stage1_cfg = config["hierarchical_svm"]["stage1"]
        stage2_cfg = config["hierarchical_svm"]["stage2"]
        self.vasc_threshold = config["hierarchical_svm"].get("vasc_threshold", 0.5)

        self.classes_ = sorted(y_train.unique().tolist())

        # --- Stage 1: vasc vs rest ---
        y1 = y_train.map(lambda l: self.vasc_label if l == self.vasc_label else "rest")
        print("Training SVM1 (vasc vs rest)...")
        self.svm1 = _train_stage(X_train, y1, groups, stage1_cfg, random_state)

        # --- Stage 2: nv vs mel (non-vasc samples only) ---
        non_vasc_mask = y_train != self.vasc_label
        X2 = X_train[non_vasc_mask]
        y2 = y_train[non_vasc_mask]
        groups2 = groups[non_vasc_mask]
        print("Training SVM2 (nv vs mel)...")
        self.svm2 = _train_stage(X2, y2, groups2, stage2_cfg, random_state)

        return self

    def predict(self, X):
        svm1_classes = list(self.svm1.classes_)
        svm1_proba = self.svm1.predict_proba(X)
        p_vasc = svm1_proba[:, svm1_classes.index(self.vasc_label)]

        # soft routing: only assign vasc if p(vasc) >= threshold
        is_vasc = p_vasc >= self.vasc_threshold
        final_pred = np.where(is_vasc, self.vasc_label, "")

        non_vasc_idx = np.where(~is_vasc)[0]
        if len(non_vasc_idx) > 0:
            X_non_vasc = X.iloc[non_vasc_idx] if hasattr(X, "iloc") else X[non_vasc_idx]
            stage2_pred = self.svm2.predict(X_non_vasc)
            final_pred[non_vasc_idx] = stage2_pred

        return final_pred

    def predict_proba(self, X):
        """
        Returns probability matrix with columns ordered by self.classes_.
        Combines SVM1 and SVM2 probabilities.
        """
        n = len(X) if not hasattr(X, "__len__") else (X.shape[0] if hasattr(X, "shape") else len(X))
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        proba = np.zeros((n, len(self.classes_)))

        # SVM1 probabilities: p(vasc) and p(rest)
        svm1_classes = list(self.svm1.classes_)
        svm1_proba = self.svm1.predict_proba(X)
        p_vasc = svm1_proba[:, svm1_classes.index(self.vasc_label)]
        p_rest = svm1_proba[:, svm1_classes.index("rest")]

        proba[:, label_to_idx[self.vasc_label]] = p_vasc

        # SVM2 probabilities: p(nv|rest) and p(mel|rest)
        svm2_proba = self.svm2.predict_proba(X)
        svm2_classes = list(self.svm2.classes_)
        for label in svm2_classes:
            if label in label_to_idx:
                proba[:, label_to_idx[label]] = p_rest * svm2_proba[:, svm2_classes.index(label)]

        return proba


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_hierarchical_svm(X_train, y_train, groups, config):
    model = HierarchicalSVM(vasc_label="vasc")
    model.fit(X_train, y_train, groups, config)
    return model


def save_model_bundle(model, feature_columns, path):
    joblib.dump({"model": model, "feature_columns": feature_columns}, path)


def load_model_bundle(path):
    return joblib.load(path)
