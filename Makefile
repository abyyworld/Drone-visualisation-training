# wildfire-watch -- development entry points.
#
# The ordering below follows the build order in README.md. Everything above
# `train` runs on a laptop with no GPU, no drone and no trained model, which
# is the property the whole repository is arranged around.

PY      ?= python3
PIP     ?= $(PY) -m pip
CONFIG  ?= config.yaml
PORT    ?= 8443
VIDEO   ?= demo/assets/wildfire_demo.mp4

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- install

.PHONY: install
install:  ## Core + dev deps. No GPU, no codecs, no ML stack.
	$(PIP) install -e ".[dev]"

.PHONY: install-station
install-station:  ## Add WebRTC, HTTPS, PyAV and OpenCV. Still no GPU needed.
	$(PIP) install -e ".[dev,stream,cv]"

.PHONY: install-inference
install-inference:  ## Add the model runtime. Install CUDA torch FIRST -- see docs/HARDWARE.md.
	$(PIP) install -e ".[inference]"

# ------------------------------------------------------------------ checks

.PHONY: lint
lint:  ## ruff
	$(PY) -m ruff check .

.PHONY: fmt
fmt:  ## ruff --fix, then report what is left
	$(PY) -m ruff check --fix .

.PHONY: test
test:  ## pytest
	$(PY) -m pytest

.PHONY: safety
safety:  ## Run only the test that enforces the safety invariants
	$(PY) -m pytest tests/test_safety_invariants.py -v

.PHONY: verify
verify: lint test  ## Everything CI runs

.PHONY: config-check
config-check:  ## Validate config.example.yaml against the schema
	$(PY) -c "from station.core.config import load_config; load_config('config.example.yaml'); print('config.example.yaml is valid')"

.PHONY: check
check:  ## Ask the station what this machine can actually do (run before an incident)
	$(PY) -m station $(if $(wildcard $(CONFIG)),-c $(CONFIG),) check

# -------------------------------------------------------------------- run

.PHONY: certs
certs:  ## Issue the station's self-signed certificate for this network
	$(PY) -m station $(if $(wildcard $(CONFIG)),-c $(CONFIG),) certs

.PHONY: run-file
run-file:  ## Step 1: the whole pipeline over a recorded clip. VIDEO=path/to.mp4
	$(PY) -m station run --source-type file --source $(VIDEO) --port $(PORT)

.PHONY: run-stub
run-stub:  ## Transport and overlay only: NO MODEL RUNS, the boxes are generated
	$(PY) -m station run --stub --source-type synthetic --port $(PORT)

.PHONY: run
run:  ## The deployment. Needs config.yaml and trained weights.
	$(PY) -m station -c $(CONFIG) run

.PHONY: replay
replay:  ## Re-serve a recorded incident. INCIDENT=incidents/<id>
	@test -n "$(INCIDENT)" || { echo "usage: make replay INCIDENT=incidents/<id>"; exit 1; }
	$(PY) -m station replay $(INCIDENT)

# ------------------------------------------------------------------- demo

.PHONY: demo
demo:  ## Render the offline overlay video (needs PyAV). VIDEO=your_footage.mp4
	$(PY) demo/render_demo.py --input $(VIDEO) --output /tmp/wildfire_demo_overlay.mp4 \
		--incident-dir /tmp/wildfire-incident-demo

.PHONY: testvideo
testvideo:  ## Generate a synthetic clip with exact ground truth
	$(PY) tools/make_test_video.py --out /tmp/wildfire_test.mp4

# --------------------------------------------------------------- training

.PHONY: datasets
datasets:  ## Merge the downloaded datasets, split by source video
	$(PY) training/prepare_datasets.py --config training/dataset_config.yaml

.PHONY: audit
audit:  ## THE GATE. Fails if the train/val split leaks. DATA=data/merged/data.yaml
	@test -n "$(DATA)" || { echo "usage: make audit DATA=data/merged/data.yaml"; exit 1; }
	$(PY) tools/audit_dataset.py --data $(DATA)

.PHONY: evaluate
evaluate:  ## Recall-first evaluation and the miss list. DATA=... WEIGHTS=...
	@test -n "$(DATA)" -a -n "$(WEIGHTS)" || { echo "usage: make evaluate DATA=data/merged/data.yaml WEIGHTS=models/x.pt"; exit 1; }
	$(PY) tools/evaluate.py --data $(DATA) --weights $(WEIGHTS)

# ------------------------------------------------------------------ chores

.PHONY: clean
clean:  ## Remove caches and build products. Never touches incidents/ or models/.
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
