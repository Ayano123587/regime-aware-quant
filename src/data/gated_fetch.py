"""Fetch pinned Bank of Canada inputs for the fixed gated forecasting workflow."""

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from src import gated_workflow as workflow

ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]
MEAN_SHA256 = "cff4c3aa15c528f308de6b7916b875ea8d60fef4c501f1bf52815f67f9e5f85e"
PARENT_SHA256 = "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e"
MEAN_ROWS = 1995
PARENT_ROWS = 2244
FIRST_DATE = "2017-01-03"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def destinations(output, protected=()):
    output = Path(output)
    sidecar = output.with_name(output.name + ".provenance.json")
    for p in (output, sidecar):
        if (
            p.exists()
            or p.is_symlink()
            or any(
                p.resolve() == Path(q).resolve()
                or p.resolve().is_relative_to(Path(q).resolve())
                for q in protected
            )
        ):
            raise ValueError("Fresh output outside immutable own artifacts required")
    return output, sidecar


def download(start, end):
    url = (
        "https://www.bankofcanada.ca/valet/observations/"
        + ",".join(ASSETS)
        + "/json?start_date="
        + start
        + "&end_date="
        + end
    )
    request = Request(
        url, headers={"User-Agent": "regime-aware-quant fixed historical research"}
    )
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    return raw, url


def observations(raw, last, count):
    rows = json.loads(raw)["observations"]
    days = [r["d"] for r in rows]
    if (
        len(rows) != count
        or days != sorted(set(days))
        or days[0] != FIRST_DATE
        or days[-1] != last
    ):
        raise ValueError("Complete pinned publication cohort required")
    for row in rows:
        if date.fromisoformat(row["d"]).isoformat() != row["d"]:
            raise ValueError("Canonical publication dates required")
        for asset in ASSETS:
            value = float(row[asset]["v"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite aligned rates required")
    return rows


def publish(output, raw, provenance, protected=()):
    output, sidecar = destinations(output, protected)
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        (sidecar, (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode()),
        (output, raw),
    ]
    # Fully written temporary files are linked exclusively; existing bytes cannot be replaced.
    for target, content in payloads:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink()
    return provenance


def provenance(url, provider_raw, published_raw, rows, mode):
    return {
        "schema": "fixed_gated_source_acquisition_v1",
        "mode": mode,
        "url": url,
        "provider_raw_sha256": hashlib.sha256(provider_raw).hexdigest(),
        "published_sha256": hashlib.sha256(published_raw).hexdigest(),
        "rows": len(rows),
        "first": rows[0]["d"],
        "last": rows[-1]["d"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Bank of Canada Valet API; no endorsement",
        "terms": "https://www.bankofcanada.ca/terms/",
        "vintage_limitation": "Current API snapshot, not historical as-published vintages",
        "provider_revision_fallback": False,
        "test_accessed": False,
    }


def mean(output):
    destinations(output)
    raw, url = download("2017-01-01", "2024-12-31")
    rows = observations(raw, "2024-12-31", MEAN_ROWS)
    from src.models.gated_forecaster import canonical_mean_bytes

    canonical = canonical_mean_bytes(rows)
    if hashlib.sha256(canonical).hexdigest() != MEAN_SHA256:
        raise ValueError(
            "Provider revision differs from pinned pre2025 snapshot; no reference fallback"
        )
    return publish(
        output, canonical, provenance(url, raw, canonical, rows, "mean_pre2025")
    )


def own_mean_freeze(run, policy, freeze_sha256, allow_different_state=False):
    run = Path(run)
    backend = workflow.production(policy)
    freeze_path = run / "mean-freeze.json"
    if sha(freeze_path) != freeze_sha256:
        raise ValueError("Reviewed own mean freeze digest required")
    freeze = json.loads(freeze_path.read_bytes())
    receipt_path = run / "mean/receipt.json"
    if (
        freeze.get("schema") != "fresh_gated_mean_freeze_v1"
        or freeze.get("policy_sha256") != workflow.POLICY_SHA256
        or freeze.get("parent_snapshot_sha256") != PARENT_SHA256
        or freeze.get("mean_source_sha256") != MEAN_SHA256
        or sha(run / "mean-source.json") != MEAN_SHA256
        or freeze.get("production_code_sha256")
        != backend.sha("src/models/gated_forecaster.py")
        or freeze.get("workflow_sha256") != sha(workflow.__file__)
        or freeze.get("calibration_started") is not False
        or freeze.get("test_accessed") is not False
        or freeze.get("reported_accuracy_reused") is not False
        or sha(receipt_path) != freeze.get("mean_receipt_sha256")
        or sha(run / "mean/bundle.joblib") != freeze.get("mean_bundle_sha256")
    ):
        raise ValueError("Own mean freeze identity or release semantics changed")
    receipt = json.loads(receipt_path.read_bytes())
    workflow.check_mean_receipt(receipt, backend)
    matches = receipt.get("mean_model_state_sha256") == workflow.REFERENCE_STATES
    if (
        receipt.get("bundle_sha256") != freeze.get("mean_bundle_sha256")
        or receipt.get("mean_model_state_sha256")
        != freeze.get("mean_model_state_sha256")
        or freeze.get("reference_numerical_state_equal") is not matches
        or (not matches and not allow_different_state)
    ):
        raise ValueError("Own model state requires explicit different-state option")
    return freeze


def calibration(run, policy, freeze_sha256, output, allow_different_state=False):
    protected = [Path(run), Path(policy)]
    destinations(output, protected)
    freeze = own_mean_freeze(run, policy, freeze_sha256, allow_different_state)
    # The first request containing 2025 observations is after authenticated mean freeze.
    raw, url = download("2017-01-01", "2025-12-31")
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise ValueError(
            "Provider revision differs from pinned complete parent; no reference fallback"
        )
    rows = observations(raw, "2025-12-31", PARENT_ROWS)
    record = provenance(url, raw, raw, rows, "calibration_parent_post_mean_freeze")
    record["mean_freeze_sha256"] = freeze_sha256
    record["mean_bundle_sha256"] = freeze["mean_bundle_sha256"]
    record["reference_numerical_state_equal"] = freeze[
        "reference_numerical_state_equal"
    ]
    return publish(output, raw, record, protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    m = sub.add_parser("mean")
    m.add_argument("--output", required=True)
    c = sub.add_parser("calibration")
    for name in ["run", "policy", "freeze-sha256", "output"]:
        c.add_argument("--" + name, required=True)
    c.add_argument("--allow-different-state", action="store_true")
    args = vars(parser.parse_args())
    action = args.pop("action")
    print(
        json.dumps({"mean": mean, "calibration": calibration}[action](**args), indent=2)
    )


if __name__ == "__main__":
    main()
