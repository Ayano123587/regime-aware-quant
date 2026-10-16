"""Strictly aligned, immutable local Bank of Canada snapshot ingestion."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_panel(panel, assets=ASSETS, max_gap=7):
    if list(panel.columns) != ["date"] + list(assets) or not len(panel):
        raise ValueError("Panel must contain date and the exact ordered asset list")
    dates = pd.to_datetime(panel.date, utc=True, errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique, finite and increasing")
    values = panel[list(assets)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("All assets need positive finite observations; no fill/drop")
    if (dates.diff().dt.total_seconds().dropna() > max_gap * 86400).any():
        raise ValueError("Shared observation gap exceeds the explicit calendar limit")
    result = panel.copy()
    result["date"] = dates
    return result


def parse_panel(content, assets=ASSETS):
    observations = json.loads(content)["observations"]
    rows = []
    for row in observations:
        if not all(asset in row and "v" in row[asset] for asset in assets):
            raise ValueError("Missing asset/date alignment; backfill is forbidden")
        rows.append({"date": row["d"], **{asset: float(row[asset]["v"]) for asset in assets}})
    return validate_panel(pd.DataFrame(rows, columns=["date"] + list(assets)), assets)


def fetch_snapshot(directory, start, end, assets=ASSETS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, receipt = directory / "source.json", directory / "provenance.json"
    url = ("https://www.bankofcanada.ca/valet/observations/" + ",".join(assets)
           + "/json?start_date=" + start + "&end_date=" + end)
    if source.exists() or receipt.exists():
        if not source.exists() or not receipt.exists():
            raise ValueError("Incomplete snapshot; preserve it and choose a fresh directory")
        provenance = json.loads(receipt.read_text(encoding="utf-8"))
        if provenance["url"] != url or sha(source) != provenance["sha256"]:
            raise ValueError("Pinned snapshot request/hash changed; no silent refresh")
        return parse_panel(source.read_bytes(), assets), provenance
    with urlopen(Request(url, headers={"User-Agent": "regime-aware-quant non-executable research"}), timeout=60) as response:
        content = response.read()
    panel = parse_panel(content, assets)
    if panel.date.min() < pd.Timestamp(start, tz="UTC") or panel.date.max() > pd.Timestamp(end, tz="UTC"):
        raise ValueError("Provider returned observations outside frozen bounds")
    source.write_bytes(content)
    panel.to_csv(directory / "panel.csv", index=False, lineterminator="\n")
    provenance = {"url": url, "sha256": sha(source), "assets": list(assets), "rows": len(panel),
                  "first": str(panel.date.min()), "last": str(panel.date.max()),
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "source": "Bank of Canada, Valet API; no endorsement",
                  "terms": "https://www.bankofcanada.ca/terms/",
                  "vintage_limitation": "Current API snapshot, not historical as-published vintages"}
    receipt.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    return panel, provenance
