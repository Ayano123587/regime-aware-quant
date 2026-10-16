"""Stage readable private source and a pinned pre-2025 development snapshot."""

import argparse
import json
from pathlib import Path
import subprocess

from src.models.interaction_study import PARENT_SHA256, check_plan, sha


def development_observations(content):
    observations = [
        r for r in json.loads(content)["observations"] if r["d"] <= "2024-12-31"
    ]
    if not observations or any(r["d"] < "2017-01-01" for r in observations):
        raise ValueError("Unexpected development observation dates")
    return observations


def prepare(parent, output, dataset, protocol="docs/interaction-plan.json"):
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise ValueError("Preserve staged payloads; choose a fresh output")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Freeze the reviewed coherent source before packaging")
    plan = check_plan(protocol)
    if sha(parent) != PARENT_SHA256:
        raise ValueError("Frozen Bank of Canada parent source changed")
    observations = development_observations(parent.read_bytes())
    files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/features/__init__.py",
        "src/data/multi_asset.py",
        "src/features/multi_asset.py",
        "src/models/regime_interactions.py",
        "src/models/interaction_study.py",
    ]
    contents = {name: Path(name).read_bytes() for name in files}
    contents["docs/interaction-plan.json"] = Path(protocol).read_bytes()
    raw = (
        json.dumps({"observations": observations}, separators=(",", ":")) + "\n"
    ).encode()
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    if digest != plan["source_derived_sha256"] or len(observations) != plan["source_rows"]:
        raise ValueError("Derived pre-2025 cohort differs from the frozen protocol")
    contents["private/development-through2024.json"] = raw
    contents["private/provenance.json"] = (
        json.dumps(
            {
                "parent_sha256": PARENT_SHA256,
                "source_sha256": digest,
                "maximum_source_date": "2024-12-31",
                "rows": len(observations),
                "first_date": observations[0]["d"],
                "last_date": observations[-1]["d"],
                "source": "Bank of Canada Valet; no endorsement",
                "terms": "https://www.bankofcanada.ca/terms/",
                "limitations": "Current snapshot rather than as-published vintage; derived before model scoring",
            },
            indent=2,
        )
        + "\n"
    ).encode()
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    for name, content in contents.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(content)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "observations_2025_or_2026_packaged": False,
    }
    (upload / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Private causal volatility interaction study",
                "licenses": [{"name": "other"}],
                "description": "Readable source and Bank of Canada pre-2025 reference observations; private research; no endorsement.",
            },
            indent=2,
        ),
        encoding="utf8",
    )
    runner = (
        Path("src/cloud/interaction_runner.py")
        .read_text(encoding="utf8")
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(upload / "source-manifest.json")),
        )
    )
    (output / "run.py").write_text(runner, encoding="utf8", newline="\n")
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "aikyatansinha/portfolio-quant-volatility-interactions-v5",
                "title": "Portfolio Quant Volatility Interactions V5",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        ),
        encoding="utf8",
    )
    (output / "preflight.json").write_text(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "manifest_sha256": sha(upload / "source-manifest.json"),
                "runner_sha256": sha(output / "run.py"),
                "source_sha256": digest,
                "rows": len(observations),
                "last_date": observations[-1]["d"],
                "expected_validation_asset_rows": 2961,
                "uploaded": False,
                "dispatched": False,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    prepare(args.parent, args.output, args.dataset)


if __name__ == "__main__":
    main()
