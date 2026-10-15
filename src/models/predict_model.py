import argparse
from pathlib import Path
import joblib
from src.data.make_dataset import load_prices
from src.features.build_features import build_features
from src.models.train_model import forecast
import pandas as pd
import numpy as np


def inference_frame(bundle, prices):
    if prices.empty or not prices.date.is_monotonic_increasing or prices.date.duplicated().any():raise ValueError("Inference prices must be nonempty, chronological and unique")
    context=bundle.get("price_context")
    if context is not None:
        overlap=context.merge(prices,on="date",suffixes=("_checkpoint","_supplied"))
        if len(overlap) and not np.allclose(overlap.close_checkpoint,overlap.close_supplied,rtol=1e-12,atol=1e-12):raise ValueError("Supplied history revises checkpoint context; refit explicitly")
        combined=pd.concat([context,prices],ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    else:combined=prices.copy()
    cutoff=pd.Timestamp(bundle["last_date"])
    if cutoff not in set(combined.date):raise ValueError("Need checkpoint cutoff and continuous rolling history")
    if combined.date.diff().dropna().dt.days.max()>7:raise ValueError("Missing bridge history: daily FX input has a gap longer than7days")
    frame=build_features(combined)
    if frame.empty or frame.date.min()>cutoff:raise ValueError("Insufficient rolling context before checkpoint cutoff")
    return frame.loc[frame.date>cutoff].copy()


def predict_prices(bundle, prices):
    frame=inference_frame(bundle,prices)
    if frame.empty:raise ValueError("No new feature rows after model training cutoff")
    predicted,posterior=forecast(bundle,frame)
    output=frame[["date"]].copy();output["predicted_next_log_return"]=predicted
    for k in range(posterior.shape[1]):output[f"regime_probability_{k}"]=posterior[:,k]
    # Bridge rows advance the filter but are not emitted unless supplied by the caller.
    return output.loc[output.date.isin(prices.date)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/run/bundle.joblib"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)  # Only load models you trained/trust.
    output = predict_prices(bundle,load_prices(args.data,min_rows=1))
    print(output.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
