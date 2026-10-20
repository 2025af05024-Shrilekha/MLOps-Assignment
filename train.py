"""Train, tune and compare two classifiers; track everything in MLflow.

Models:  Logistic Regression  vs  Random Forest
Tuning:  GridSearchCV, stratified 5-fold, optimising ROC-AUC
Output:  models/model.joblib (best pipeline) + models/model_meta.json
         MLflow runs: params, CV + test metrics, plots, model artifact

Run:  python -m heartml.train
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from heartml import config  # noqa: E402
from heartml.features import build_pipeline  # noqa: E402

log = logging.getLogger("heartml.train")

SCORING = {
    "accuracy": "accuracy", "precision": "precision", "recall": "recall",
    "f1": "f1", "roc_auc": "roc_auc",
}

# name -> (estimator, param grid). Grid keys are prefixed with the pipeline step.
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=config.RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 0.3, 1.0, 3.0], "clf__penalty": ["l1", "l2"],
         "clf__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 4, 6],
         "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]},
    ),
}


def load_xy(path=None):
    df = pd.read_csv(path or config.CLEAN_FILE)
    return df[config.FEATURES], df[config.TARGET]


def test_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def _feature_names(model) -> list[str]:
    return list(model.named_steps["prep"].get_feature_names_out())


def _plots(name: str, model, X_test, y_test, out_dir) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}: confusion matrix")
    p = out_dir / f"{name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{name}: ROC (test)")
    p = out_dir / f"{name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    clf = model.named_steps["clf"]
    weights = getattr(clf, "feature_importances_", None)
    label = "importance"
    if weights is None:
        weights, label = clf.coef_.ravel(), "coefficient"
    s = pd.Series(weights, index=_feature_names(model)).sort_values(key=np.abs).tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(s.index, s.values, color=["#C8553D" if v > 0 else "#4C9F70" for v in s.values])
    ax.set_title(f"{name}: top features ({label})")
    p = out_dir / f"{name}_feature_{label}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return [str(x) for x in paths]


def train_one(name, estimator, grid, X_train, y_train, X_test, y_test, cv) -> dict:
    with mlflow.start_run(run_name=name) as run:
        search = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc", cv=cv,
                              n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Unbiased-ish CV estimate of the tuned model on all metrics
        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
        cv_metrics = {f"cv_{k}_mean": float(np.mean(cv_res[f"test_{k}"])) for k in SCORING}
        cv_metrics.update({f"cv_{k}_std": float(np.std(cv_res[f"test_{k}"])) for k in SCORING})
        t_metrics = test_metrics(best, X_test, y_test)

        mlflow.set_tags({"model_family": name, "sklearn": sklearn.__version__,
                         "python": platform.python_version()})
        mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_params({"cv_folds": config.CV_FOLDS, "test_size": config.TEST_SIZE,
                           "grid_size": len(search.cv_results_["params"]),
                           "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({**cv_metrics, **t_metrics, "grid_best_cv_roc_auc": search.best_score_})

        out_dir = config.REPORTS_DIR / "runs" / name
        for p in _plots(name, best, X_test, y_test, out_dir):
            mlflow.log_artifact(p, artifact_path="plots")
        pd.DataFrame(search.cv_results_).to_csv(out_dir / "grid_search_results.csv", index=False)
        mlflow.log_artifact(str(out_dir / "grid_search_results.csv"), artifact_path="tuning")
        report = classification_report(y_test, best.predict(X_test), output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        signature = infer_signature(X_train, best.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_train.head(3))

        log.info("%-20s best=%s  cv_auc=%.4f  test_auc=%.4f", name, search.best_params_,
                 cv_metrics["cv_roc_auc_mean"], t_metrics["test_roc_auc"])
        return {"name": name, "run_id": run.info.run_id, "model": best,
                "best_params": search.best_params_, "cv": cv_metrics, "test": t_metrics}


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    results = [train_one(n, est, grid, X_train, y_train, X_test, y_test, cv)
               for n, (est, grid) in CANDIDATES.items()]

    # Select on CV ROC-AUC (training data only) - the test set stays a final check.
    best = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], config.MODEL_FILE)

    meta = {
        "model_name": best["name"], "mlflow_run_id": best["run_id"],
        "best_params": best["best_params"], "selection_metric": "cv_roc_auc_mean",
        "cv_metrics": best["cv"], "test_metrics": best["test"],
        "features": config.FEATURES, "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison": {r["name"]: {"cv_roc_auc": round(r["cv"]["cv_roc_auc_mean"], 4),
                                   "test_roc_auc": round(r["test"]["test_roc_auc"], 4),
                                   "test_accuracy": round(r["test"]["test_accuracy"], 4)}
                       for r in results},
    }
    config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_as_best", "true")
        mlflow.log_artifact(str(config.MODEL_FILE), artifact_path="joblib")

    log.info("Best model: %s -> %s", best["name"], config.MODEL_FILE)
    log.info("Comparison: %s", json.dumps(meta["comparison"]))
    return meta


if __name__ == "__main__":
    main()
