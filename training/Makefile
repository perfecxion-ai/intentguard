# Training pipeline orchestration.
#
# Usage:
#   make train-vertical VERTICAL=finance PROVIDER=anthropic
#   make generate VERTICAL=finance
#   make train VERTICAL=finance
#   make calibrate VERTICAL=finance
#   make export VERTICAL=finance

VERTICAL ?= finance
PROVIDER ?= anthropic
DATA_DIR = data/$(VERTICAL)
MODEL_DIR = models/$(VERTICAL)
DIST_DIR = dist/$(VERTICAL)
POLICY = policies/$(VERTICAL).json

.PHONY: generate mine train calibrate export train-vertical clean

# Full pipeline
train-vertical: generate mine train calibrate export
	@echo "Pipeline complete for $(VERTICAL)"
	@echo "Artifacts in $(DIST_DIR)/"

# Generate synthetic training data
generate:
	@mkdir -p $(DATA_DIR)
	python -m training.data_generation.synthetic_generator \
		--policy $(POLICY) \
		--output $(DATA_DIR)/synthetic.jsonl \
		--provider $(PROVIDER) \
		--count 2000

# Mine hard negatives from existing data
mine:
	python -m training.data_generation.hard_negative_miner \
		--input $(DATA_DIR)/synthetic.jsonl \
		--policy $(POLICY) \
		--output $(DATA_DIR)/hard_negatives.jsonl \
		--provider $(PROVIDER)

# Fine-tune the model
train:
	python -m training.fine_tune \
		--data $(DATA_DIR) \
		--config training/train_config.yaml \
		--output $(MODEL_DIR)

# Calibrate with temperature scaling
calibrate:
	python -m training.calibrate \
		--model $(MODEL_DIR)/best \
		--data $(DATA_DIR)/synthetic.jsonl \
		--output $(MODEL_DIR)/calibration_params.json \
		--plot $(MODEL_DIR)/reliability_diagram.png

# Export to ONNX with sanity gate
export:
	python -m training.export_onnx \
		--model $(MODEL_DIR)/best \
		--output $(DIST_DIR) \
		--calibration $(MODEL_DIR)/calibration_params.json \
		--sanity-data $(DATA_DIR)/synthetic.jsonl \
		--sanity-count 100

# Clean artifacts for a vertical
clean:
	rm -rf $(DATA_DIR) $(MODEL_DIR) $(DIST_DIR)
