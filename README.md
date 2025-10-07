# ProjectBinaryClassification
End-to-end пайплайн бинарной классификации на датасете Cardiovascular Disease (70k записей, Kaggle)
# Project structure
heart_project/
├─ src/heart_pred/             
│  ├─ __init__.py
│  ├─ data_preprocessing.py
│  ├─ model.py
│  └─ utils.py
├─ scripts/
│  ├─ train.py
│  └─ evaluate.py
├─ requirements.txt
└─ tests/
   └─ test_preprocessing.py

# Quickstart

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training:

```bash
python scripts/train.py --data path/to/your/heart.csv --target target --output-dir artifacts
```

3. Evaluate:

```bash
python scripts/evaluate.py --data path/to/your/heart.csv --model-path artifacts/model.joblib
```

# What's included

- Refactored code split into `data_preprocessing`, `model`, and `utils`.
- Simple training and evaluation scripts.
- `requirements.txt` with core dependencies.

# Next steps (suggested)

- Add unit tests and CI (a starter `tests/` is included).
- Add dvc pipeline for data and model versioning.
- Add linters (`black`, `flake8`, `isort`) and pre-commit hooks.
- Improve configuration management (e.g., `hydra`, `yaml` configs).
