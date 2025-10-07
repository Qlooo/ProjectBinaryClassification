# ProjectBinaryClassification
End-to-end пайплайн бинарной классификации на датасете Cardiovascular Disease (70k записей, Kaggle)
# Структура проекта

```
heart_project/
│
├── src/heart_pred/              # Исходный код проекта
│   ├── data_preprocessing.py    # Предобработка данных
│   ├── model.py                 # Обучение и оценка модели
│   └── utils.py                 # Утилиты (сохранение, загрузка и т.д.)
│
├── scripts/                     # Скрипты для пайплайна
│   ├── preprocess.py            # Очистка и подготовка данных
│   ├── train.py                 # Обучение модели
│   └── evaluate.py              # Оценка модели
│
├── data/
│   ├── raw/                     # Исходные (сырые) данные
│   └── processed/               # Обработанные данные
│
├── artifacts/                   # Сохранённые модели и результаты
│
├── tests/                       # Модульные тесты (pytest)
│   └── test_preprocessing.py
│
├── notebooks/                   # Исходный ноутбук
│   └── Heart_prediction.ipynb
│
├── params.yaml                  # Параметры обучения (размер теста, random_state, модель и т.д.)
├── dvc.yaml                     # DVC-пайплайн (preprocess → train → evaluate)
├── requirements.txt             # Основные зависимости
├── requirements-dev.txt         # Зависимости для разработки (линтеры, тесты, DVC)
├── Makefile                     # Команды для автоматизации
├── .pre-commit-config.yaml      # Конфигурация хуков pre-commit
├── .github/workflows/ci.yml     # GitHub Actions CI
└── README.md                    # Описание проекта
```

---

# Установка и запуск

### 1. Создание виртуального окружения
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

### 2. Предобработка и обучение модели вручную
```bash
python scripts/preprocess.py --input-path data/raw/heart.csv --output-path data/processed/heart_processed.csv
python scripts/train.py --data data/processed/heart_processed.csv --target target --output-dir artifacts
python scripts/evaluate.py --data data/processed/heart_processed.csv --model-path artifacts/model.joblib
```

---

# Тестирование
Для запуска модульных тестов:
```bash
pytest -v
```

---

# Контроль качества кода

### Установка и запуск линтеров
```bash
pre-commit install
pre-commit run --all-files
```

### Используемые инструменты
- **black** — автоформатирование кода  
- **isort** — сортировка импортов  
- **flake8** — анализ качества и стиля кода  
- **pytest** — тестирование  

Конфигурации линтеров хранятся в файлах:
- `.pre-commit-config.yaml`
- `.flake8`
- `pyproject.toml`

---

# CI/CD (GitHub Actions)
Репозиторий содержит workflow `.github/workflows/ci.yml`,  
который автоматически:
1. Устанавливает зависимости  
2. Проверяет стиль кода (`flake8`, `black`, `isort`)  
3. Запускает тесты (`pytest`)  
4. Работает при каждом **push** и **pull request** в ветки `main` / `master`

---

# DVC (Data Version Control)

Проект поддерживает систему **DVC** для воспроизводимых экспериментов.  

### Инициализация и запуск пайплайна:
```bash
dvc init
dvc add data/raw/heart.csv
dvc repro
```

Пайплайн состоит из этапов:
1. `preprocess` — очистка и подготовка данных  
2. `train` — обучение модели  
3. `evaluate` — оценка точности  

Результаты сохраняются в `data/processed/` и `artifacts/`.

---

# Используемые библиотеки
- **pandas**, **numpy** — работа с данными  
- **scikit-learn**, **xgboost**, **lightgbm** — обучение моделей  
- **matplotlib**, **seaborn** — визуализация  
- **joblib** — сериализация моделей  
- **dvc**, **pytest**, **black**, **flake8**, **isort** — автоматизация и тестирование  

---
