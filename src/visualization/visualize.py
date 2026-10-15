import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("models/run/predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/equity.png"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    fig, ax = plt.subplots(figsize=(9, 4))
    for column in ["regime_prediction", "baseline_prediction"]:
        position = np.sign(frame[column])
        net = position*frame.target - np.abs(np.diff(position, prepend=0))*.0002
        ax.plot(pd.to_datetime(frame.date), np.exp(np.cumsum(net)), label=column)
    ax.plot(pd.to_datetime(frame.date), np.exp(frame.target.cumsum()), label="price-index buy/hold")
    ax.set(ylabel="Simulated equity", title="Chronological out-of-sample diagnostic (2 bps turnover cost)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
