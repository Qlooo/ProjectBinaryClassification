
"""Utility functions for ML project."""
import pandas as pd
import numpy as np

def save_model_pickle(model, path: str):
    import joblib
    joblib.dump(model, path)

def load_model_pickle(path: str):
    import joblib
    return joblib.load(path)
