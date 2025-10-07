
.PHONY: venv install train test

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

train:
	python scripts/train.py --data data/heart.csv --target target --output-dir artifacts

test:
	pytest -q
