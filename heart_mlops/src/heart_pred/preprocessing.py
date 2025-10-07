from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # object/category -> categorical; остальные -> numeric
    categorical: List[str] = [c for c in X.columns if X[c].dtype.name in ("object", "category")]
    numeric: List[str] = [c for c in X.columns if c not in categorical]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        # Совместимость с разными версиями sklearn:
        # >=1.2: OneHotEncoder(..., sparse_output=False)
        # <1.2:  OneHotEncoder(..., sparse=False)
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
