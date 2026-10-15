.PHONY: install data train predict test plot
install:
	python -m pip install -e ".[dev]"
data:
	python -m src.data.make_dataset
train:
	python -m src.models.train_model
predict:
	python -m src.models.predict_model --data data/raw/prices.csv
test:
	python -m pytest -q
plot:
	python -m src.visualization.visualize
