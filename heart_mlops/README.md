# Heart Prediction — Reproducible ML Pipeline

**Role focus:** ML/DevOps engineering — code engineering, reproducibility, automation, documentation, deployment preparedness.

This repository converts a research notebook into a clean, testable, and reproducible project. It includes:
- Structured Python package (`src/heart_pred`)
- Versioned environments (`requirements.txt`, `environment.yml`)
- Refactored code split into logical modules (data, preprocessing, training, evaluation)
- Unit tests (pytest)
- Code quality (black, isort, flake8)
- One-command reproducible pipeline with DVC (`dvc.yaml`, `params.yaml`)
- Clear instructions in this README

> **Notebook provenance**: This project was bootstrapped from `Heart_prediction.ipynb`.  
> Short excerpt / signals detected to infer target column and structure:

```
**Cardiovascular Disease (сердечно-сосудистые заболевания)** — публичный набор данных (~70 000 наблюдений), цель — бинарная переменная cardio (1 — есть заболевание, 0 — нет). Каждая строка — один пациент/осмотр.

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

**Основные признаки**:

*   **age** — возраст в днях;
*   **gender** — пол (1 — женщина, 2 — мужчина);
*   **height** — рост, см;
*   **weight** — масса, кг;
*   **ap_hi** — систолическое артериальное давление, мм рт. ст.;
*   **ap_lo** — диастолическое артериальное давление, мм рт. ст.;
*   **cholesterol** — ур
```

Target column inferred: `num` (you can change it in `params.yaml`).

---

## Quickstart

### 1) Clone and setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Or with Conda:
```bash
conda env create -f environment.yml
conda activate heart-mlops
```

### 2) Put your data
- If you have a CSV locally, put it at `data/heart.csv` (default) or update `params.yaml:data.path`.
- If you have a URL, set `params.yaml:data.url` and DVC will download it.

### 3) Reproduce the full pipeline
```bash
dvc repro
```
Artifacts:
- `artifacts/model.joblib` — trained pipeline (preprocessing + model)
- `artifacts/metrics.json` — holdout and CV metrics

### 4) Run tests and linters
```bash
pytest
isort src tests
black src tests
flake8 src tests
```

### 5) Train & evaluate manually
```bash
python -m heart_pred.train --params params.yaml --artifacts artifacts
python -m heart_pred.evaluate --model artifacts/model.joblib --params params.yaml
```

---

## Project structure

```
.
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Makefile
├── README.md
├── src/
│   └── heart_pred/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── train.py
│       └── evaluate.py
├── tests/
│   └── test_preprocessing.py
└── data/            # (ignored) place heart.csv here or set a URL in params.yaml
```

---

## Notes on deployment preparedness

- The pipeline saves a single `joblib` artifact that includes preprocessing and the model; it can be loaded directly in an API service to serve predictions.
- To containerize, add a minimal `Dockerfile` and `uvicorn` FastAPI app that `joblib.load`s the model (out of scope here but trivial to add).
- DVC remotes (e.g., S3, GDrive) can be set with `dvc remote add -d <name> <url>` to version data and artifacts alongside code.

---

## Maintainer commands

- `make all` — format, lint, test, train, evaluate
- `make pipeline` — run the DVC pipeline

---

## License

MIT (or your preference).
