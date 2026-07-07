# Linux / Orange Pi task runner. Mirrors tasks.ps1.
#
# Usage:
#   make setup
#   make test
#   make replay CLIP=verification/fixtures/my-clip
#   make run-dry
#   make run-live
#   make train DATA=training/data/data.yaml
#   make validate WEIGHTS=models/best.pt DATA=training/data/data.yaml

PYTHON ?= python3.11
VENV   ?= .venv
PIP     = $(VENV)/bin/pip
PY      = $(VENV)/bin/python
CLI     = $(VENV)/bin/spaghetti-guard

.PHONY: setup test lint replay run-dry run-live train validate clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src verification tests training

replay:
	@test -n "$(CLIP)" || (echo "usage: make replay CLIP=<path>" && exit 2)
	$(CLI) replay $(CLIP) $(EXTRA)

run-dry:
	$(CLI) run --dry-run $(EXTRA)

run-live:
	@echo "Live mode — guard will publish real stop/pause commands."
	$(CLI) run $(EXTRA)

train:
	@test -n "$(DATA)" || (echo "usage: make train DATA=<data.yaml>" && exit 2)
	$(PY) training/train.py --data $(DATA) $(EXTRA)

validate:
	@test -n "$(WEIGHTS)" || (echo "usage: make validate WEIGHTS=<.pt> DATA=<data.yaml>" && exit 2)
	@test -n "$(DATA)" || (echo "usage: make validate WEIGHTS=<.pt> DATA=<data.yaml>" && exit 2)
	$(PY) training/validate.py --weights $(WEIGHTS) --data $(DATA) $(EXTRA)

# NOTE: deliberately does NOT remove runs/ — it holds training results.
clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	rm -rf build dist *.egg-info
