
import pandas as pd
import numpy as np
from src.heart_pred.data_preprocessing import basic_cleaning, split_features_target

def test_basic_cleaning_and_split():
    df = pd.DataFrame({
        'a': [1, 2, None, 2],
        'b': ['x', None, 'y', 'y'],
        'target': [0, 1, 0, 1]
    })
    df_clean = basic_cleaning(df)
    assert df_clean.isnull().sum().sum() == 0
    X, y = split_features_target(df_clean, target='target')
    assert 'target' not in X.columns
    assert len(y) == len(df_clean)
