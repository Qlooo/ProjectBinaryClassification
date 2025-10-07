# Heart Prediction — воспроизводимый ML-проект

**Фокус роли:** ML/DevOps-инженерия — инженерия кода, воспроизводимость, автоматизация, документация, готовность к деплою.

Репозиторий конвертирует исследовательский ноутбук в чистый, тестируемый и воспроизводимый проект. В составе:
- Структурированный Python‑пакет `src/heart_pred`
- Зафиксированное окружение (`requirements.txt`, `environment.yml`)
- Рефакторинг на логические модули (данные, предобработка, обучение, оценка)
- Модульные тесты (pytest)
- Контроль качества кода (black, isort, flake8)
- Воспроизводимый конвейер с DVC (`dvc.yaml`, `params.yaml`)
- Пошаговые инструкции (этот README)

> **Происхождение:** проект собран из ноутбука `Heart_prediction.ipynb`.  
> Целевая колонка настраивается в `params.yaml` (проверь свой датасет — обычно `cardio` или `num`).

---

## Быстрый старт

### 1) Подготовка окружения
**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### 2) Данные
- Положи CSV в `data/heart.csv` **или** укажи путь/URL в `params.yaml`:
  ```yaml
  data:
    url: ""            # оставь пустым, если файл локальный
    path: data/heart.csv
    target: cardio     # или твой таргет
    test_size: 0.2
  ```
Загрузчик автоматически определяет разделитель (`,`/`;`/`\t`).

### 3) Полный конвейер (DVC)
```bash
# если DVC ещё не инициализирован
dvc init --no-scm   # или просто `dvc init`, если используешь git
dvc repro           # запустить все стадии (train -> evaluate)
```
Артефакты:
- `artifacts/model.joblib` — конвейер (предобработка + модель)
- `artifacts/metrics.json` — метрики валидации/holdout

### 4) Ручной запуск (без DVC)
```bash
export PYTHONPATH=$PWD/src   # Windows: set PYTHONPATH=%cd%\src
python -m heart_pred.train --params params.yaml --artifacts artifacts
python -m heart_pred.evaluate --model artifacts/model.joblib --params params.yaml
```

### 5) Тесты и линтинг
```bash
pytest
isort src tests
black src tests
flake8 src tests
```

---

## Структура проекта

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
│       ├── data.py            # загрузка и автодетект разделителя
│       ├── preprocessing.py   # разбиение X/y, пайплайн предобработки
│       ├── train.py           # обучение + CV + сохранение артефактов
│       └── evaluate.py        # оценка модели на данных
├── tests/
│   └── test_preprocessing.py
└── data/                      # .gitignore/.dvcignore: положи сюда heart.csv
```

---

## Модель

Модель задаётся параметрами в `params.yaml`. По умолчанию — логистическая регрессия. Можно переключить на XGBoost (пример):

```yaml
model:
  type: xgboost
  xgboost:
    n_estimators: 400
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.9
    colsample_bytree: 0.9
    reg_lambda: 1.0
    tree_method: auto
```
> Для XGBoost добавь в `requirements.txt`: `xgboost>=2.0`, затем `pip install -r requirements.txt`.

---

## Визуализация результатов (ROC/PR/матрица ошибок/важность признаков)

Скрипт `scripts/visualize.py` строит ключевые графики и сохраняет их в `reports/`:
```bash
export PYTHONPATH=$PWD/src
python scripts/visualize.py --model artifacts/model.joblib --params params.yaml --outdir reports
# macOS:
open reports/roc_curve.png reports/pr_curve.png reports/confusion_matrix.png reports/feature_importance.png
```

Чтобы запускать одной командой — добавь цель в `Makefile`:
```make
report:
	PYTHONPATH=src python scripts/visualize.py --model artifacts/model.joblib --params params.yaml --outdir reports
```

И/или оформи отдельной стадией в `dvc.yaml` после `evaluate`.

---

## Команды сопровождающего (maintainer)

- `make all` — форматирование, линтинг, тесты, обучение, оценка  
- `make report` — генерация графиков (если добавлен `scripts/visualize.py`)  
- `dvc repro` — полный воспроизводимый прогон конвейера  
- `dvc remote add -d <name> <url>` → `dvc push` — подключение и выгрузка артефактов в удалённое хранилище (S3/GDrive/WebDAV и т.д.)

---

