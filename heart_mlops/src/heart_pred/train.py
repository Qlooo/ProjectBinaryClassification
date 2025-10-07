from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from heart_pred.data import load_csv, maybe_download
from heart_pred.preprocessing import build_preprocessor, split_features_target


def get_model(params: dict) -> LogisticRegression:
    mcfg = params.get("model", {}).get("logreg", {})
    return LogisticRegression(
        C=mcfg.get("C", 1.0),
        max_iter=mcfg.get("max_iter", 2000),
        class_weight=mcfg.get("class_weight", "balanced"),
        solver="liblinear",
        random_state=params.get("seed", 42),
    )


def main(params_path: str, artifacts_dir: str = "artifacts") -> None:
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    os.makedirs(artifacts_dir, exist_ok=True)

    # Data
    data_cfg = params.get("data", {})
    csv_path = data_cfg.get("path", "data/heart.csv")
    url = data_cfg.get("url", "")
    if url:
        csv_path = maybe_download(url, csv_path)

    df = load_csv(csv_path)
    X, y = split_features_target(df, target=data_cfg.get("target", "target"))

    # Preprocess + model pipeline
    preprocessor = build_preprocessor(X)
    model = get_model(params)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_cfg.get("test_size", 0.2)),
        stratify=y,
        random_state=params.get("seed", 42),
    )

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.get("seed", 42))
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit final
    pipe.fit(X_train, y_train)

    # Evaluate on holdout
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        "cv_mean": {k: float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")},
    }

    # Save artifacts
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(pipe, model_path)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"model_path": model_path, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts output dir")
    args = parser.parse_args()
    main(args.params, args.artifacts)
