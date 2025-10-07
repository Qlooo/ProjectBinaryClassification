from __future__ import annotations

import os
import urllib.request

import pandas as pd


def maybe_download(url: str, out_path: str) -> str:
    """Download CSV from URL if provided; otherwise, just return out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if url:
        urllib.request.urlretrieve(url, out_path)
    return out_path


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    # auto-detect delimiter (handles ',', ';', '\t', etc.)
    return pd.read_csv(path, sep=None, engine="python")
