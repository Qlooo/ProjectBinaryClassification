
"""Model training and evaluation functions."""
from typing import Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def get_model(name: str = 'logistic', **kwargs) -> Any:
    if name == 'logistic':
        return LogisticRegression(max_iter=1000, **kwargs)
    elif name == 'rf' or name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y, preds)
    roc = roc_auc_score(y, probs) if probs is not None else None
    report = classification_report(y, preds, digits=4)
    cm = confusion_matrix(y, preds)
    return {'accuracy': acc, 'roc_auc': roc, 'report': report, 'confusion_matrix': cm}
