from __future__ import annotations
import argparse, json, os, joblib, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def load_data(params_path: str):
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    path = params["data"]["path"]
    target = params["data"]["target"]
    df = pd.read_csv(path, sep=None, engine="python")  # авто-детект разделителя
    y = df[target]
    drop_cols = [c for c in ["id", target] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y

def main(model_path: str, params_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    X, y = load_data(params_path)
    pipe = joblib.load(model_path)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        s = pipe.decision_function(X)
        proba = (s - s.min()) / (s.max() - s.min() + 1e-12)
    pred = (proba >= 0.5).astype(int)

    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curve"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150); plt.close()

    # Feature importance (если есть)
    model = pipe.named_steps.get("model")
    names = np.array(X.columns)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_.ravel())
    else:
        imp = None
    if imp is not None:
        order = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(8, 6))
        y_pos = np.arange(len(order))
        plt.barh(y_pos, imp[order]); plt.yticks(y_pos, names[order]); plt.gca().invert_yaxis()
        plt.xlabel("Importance"); plt.title("Top feature importances / coefficients")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150); plt.close()

    with open(os.path.join(outdir, "viz_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": float((pred==y).mean()), "roc_auc": float(roc_auc), "n": int(len(y))}, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--outdir", default="reports")
    a = p.parse_args()
    main(a.model, a.params, a.outdir)
