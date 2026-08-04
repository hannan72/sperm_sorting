# sperm_sorting -- development shortcuts.
.DEFAULT_GOAL := help
PY ?= .venv/bin/python
PIP ?= $(PY) -m pip

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------- setup
.PHONY: venv install install-dev
venv:  ## Create the virtualenv
	python3 -m venv .venv && $(PIP) install -U pip

install:  ## Install the package
	$(PIP) install -e .

install-dev:  ## Install with every optional extra
	$(PIP) install -e ".[all]"

.PHONY: lock
lock:  ## Freeze the current environment into requirements.lock
	$(PIP) freeze --exclude-editable > requirements.lock
	@echo "wrote requirements.lock"

# --------------------------------------------------------------- quality
.PHONY: lint format typecheck check
lint:  ## Lint with ruff
	$(PY) -m ruff check src datasets training tests web scripts

format:  ## Auto-format and fix with ruff
	$(PY) -m ruff format src datasets training tests web scripts
	$(PY) -m ruff check --fix src datasets training tests web scripts

typecheck:  ## Type-check with mypy
	$(PY) -m mypy src

check: lint typecheck test  ## Lint, type-check and test

# --------------------------------------------------------------- tests
.PHONY: test test-fast test-cov
test:  ## Run the whole suite
	$(PY) -m pytest -q

test-fast:  ## Skip the slow end-to-end tests
	$(PY) -m pytest -q -m "not slow"

test-cov:  ## Run with coverage
	$(PY) -m pytest -q --cov=src/sperm_sorting --cov-report=term-missing

# --------------------------------------------------------------- run
.PHONY: doctor feasibility run-synthetic demo
doctor:  ## Report environment, calibration and model status
	$(PY) -m sperm_sorting.cli doctor

feasibility:  ## Check the optical and throughput budget
	$(PY) -m sperm_sorting.cli feasibility -c configs/synthetic.yaml

run-synthetic:  ## Run the pipeline against the simulator
	$(PY) -m sperm_sorting.cli run -c configs/synthetic.yaml -n 500

demo:  ## Serve the interactive web demo
	$(PY) -m uvicorn web.app:app --reload --port 8000

# --------------------------------------------------------------- data
.PHONY: generate-data
generate-data:  ## Build a synthetic morphology dataset
	$(PY) -m sperm_sorting.simulator.generate --n 20000 --out data/ --image-size 128

# --------------------------------------------------------------- training
.PHONY: train-morphology train-detector eval-pipeline
train-morphology:  ## Train the four-head morphology model
	$(PY) training/train_morphology.py --config configs/training/morphology.yaml

train-detector:  ## Train the detector
	$(PY) training/train_detector.py --config configs/training/detector.yaml

eval-pipeline:  ## Measure the end-to-end product against ground truth
	$(PY) training/eval_pipeline.py --config configs/synthetic.yaml

# --------------------------------------------------------------- docker
.PHONY: docker-build docker-run
docker-build:  ## Build the container image
	docker build -t sperm-sorting:latest .

docker-run:  ## Run the synthetic pipeline in the container
	docker run --rm -it sperm-sorting:latest run -c configs/synthetic.yaml -n 200

# --------------------------------------------------------------- cleanup
.PHONY: clean
clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage
