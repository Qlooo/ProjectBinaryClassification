import pandas as pd
import pytest

from heart_pred.preprocessing import build_preprocessor, split_features_target


def test_split_features_target_ok():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0], "target": [1, 0, 1]})
    X, y = split_features_target(df, target="target")
    assert "target" not in X.columns
    assert y.shape[0] == df.shape[0]


def test_split_features_target_missing():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    with pytest.raises(KeyError):
        split_features_target(df, target="target")


def test_build_preprocessor_shapes():
    df = pd.DataFrame({"age": [50, 60, 70], "sex": ["M", "F", "M"], "chol": [200, 180, 220]})
    pre = build_preprocessor(df)
    Xt = pre.fit_transform(df)
    assert Xt.shape[0] == 3
