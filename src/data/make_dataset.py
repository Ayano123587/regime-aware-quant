"""Fetch a fixed historical range; keep source bytes and provenance locally."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

URL = "https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date=2017-01-01&end_date=2025-12-31"


def load_prices(path, min_rows=150):
    frame = pd.read_csv(path)
    if not {"date", "close"} <= set(frame):
        if len(frame.columns) != 2:
            raise ValueError("Expected date,close CSV or two-column FRED CSV")
        frame.columns = ["date", "close"]
    frame = frame[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise", utc=True)
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Missing or malformed dates/prices are not silently dropped")
    if not frame.date.is_monotonic_increasing:
        raise ValueError("Price dates must be strictly chronological")
    frame = frame.reset_index(drop=True)
    if frame.date.duplicated().any() or (frame.close <= 0).any() or len(frame) < min_rows:
        raise ValueError(f"Prices require >={min_rows} unique dates and positive finite closes")
    import numpy as np
    if not np.isfinite(frame.close).all():
        raise ValueError("Prices must be finite")
    return frame


def fetch(output=Path("data/raw")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(URL, headers={"User-Agent": "regime-aware-quant research"}), timeout=30) as response:
        content = response.read()
    raw = output / "boc-usdcad.json"
    raw.write_bytes(content)
    observations = json.loads(content)["observations"]
    frame = pd.DataFrame([{"date": row["d"], "close": row["FXUSDCAD"]["v"]}
                          for row in observations if "FXUSDCAD" in row])
    frame.to_csv(output / "prices.csv", index=False)
    frame = load_prices(output / "prices.csv")
    frame.to_csv(output / "prices.csv", index=False)
    metadata = {"url": URL, "sha256": hashlib.sha256(content).hexdigest(), "rows": len(frame),
                "first": str(frame.date.min()), "last": str(frame.date.max()),
                "citation": "Bank of Canada, Valet API, FXUSDCAD daily exchange rates",
                "terms": "https://www.bankofcanada.ca/terms/; retain source attribution; no Bank endorsement"}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    print(json.dumps(fetch(parser.parse_args().output), indent=2))
