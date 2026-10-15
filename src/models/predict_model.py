import argparse
from pathlib import Path
import joblib
from src.data.make_dataset import load_prices
from src.features.build_features import build_features
from src.models.train_model import forecast
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/run/bundle.joblib"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)  # Only load models you trained/trust.
    frame = build_features(load_prices(args.data))
    frame = frame.loc[frame.date > pd.Timestamp(bundle["last_date"])]
    if frame.empty:
        raise SystemExit("No new feature rows after model training cutoff")
    predicted, posterior = forecast(bundle, frame)
    output = frame[["date"]].copy()
    output["predicted_next_log_return"] = predicted
    for k in range(posterior.shape[1]):
        output[f"regime_probability_{k}"] = posterior[:, k]
    print(output.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
