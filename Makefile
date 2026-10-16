.PHONY: install data train calibration-data calibrate predict test plot advanced-search advanced-test advanced-report
POLICY ?= docs/gated-forecast-policy.json
MEAN_SNAPSHOT ?= data/processed/development-through2024.json
DEVELOPMENT_PARENT ?= data/external/gated-development/source.json
MEAN_RUN ?= models/my-gated-mean
CALIBRATION_RUN ?= models/my-gated-calibration
FORECAST_SOURCE ?= data/new-observations.json
FORECAST_OUTPUT ?= data/reports/my-forecasts.csv
FREEZE_SHA ?=
RECEIPT_SHA ?=
POLICY_SHA ?= 0f511814a9cd0e4f97626bdd8deafbe3d290b25b7829e210eeed6495c4339bd0

install:
	python -m pip install -e ".[dev]"
data:
	python -m src.data.gated_fetch mean --output "$(MEAN_SNAPSHOT)"
train:
	python train.py selected --snapshot "$(MEAN_SNAPSHOT)" --policy "$(POLICY)" --output "$(MEAN_RUN)"
calibration-data:
	python -c "import sys; assert len(sys.argv[1]) == 64, 'Set FREEZE_SHA to the reviewed own mean-freeze digest'" "$(FREEZE_SHA)"
	python -m src.data.gated_fetch calibration --run "$(MEAN_RUN)" --policy "$(POLICY)" --freeze-sha256 "$(FREEZE_SHA)" --output "$(DEVELOPMENT_PARENT)"
calibrate:
	python -m src.gated_workflow calibrate --run "$(MEAN_RUN)" --parent "$(DEVELOPMENT_PARENT)" --policy "$(POLICY)" --freeze-sha256 "$(FREEZE_SHA)" --output "$(CALIBRATION_RUN)"
predict:
	python -m src.gated_workflow forecast --model "$(CALIBRATION_RUN)/calibrated" --snapshot "$(FORECAST_SOURCE)" --policy "$(POLICY)" --policy-sha256 "$(POLICY_SHA)" --receipt-sha256 "$(RECEIPT_SHA)" --output "$(FORECAST_OUTPUT)"
test:
	python -m pytest -q
plot:
	python -m src.visualization.visualize
advanced-search:
	python -m src.models.advanced search
advanced-test:
	python -m src.models.advanced test
advanced-report:
	python -m src.models.advanced report
