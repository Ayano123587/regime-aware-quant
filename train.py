"""Train the selected causal FX mean on a pinned historical snapshot."""

import argparse
from pathlib import Path


def create_run(output, plan):
    output = Path(output)
    if output.exists():
        raise ValueError(
            "Output already exists; preserve historical artifacts and choose a fresh directory"
        )
    content = Path(plan).read_bytes()
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes(content)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["selected", "shared", "blend", "recency"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--policy", default="docs/gated-forecast-policy.json")
    parser.add_argument("--allow-different-state", action="store_true")
    args = parser.parse_args(argv)
    if args.study == "selected":
        if args.snapshot is None:
            parser.error(
                "selected requires --snapshot with the pinned canonical pre2025 input"
            )
        from src.gated_workflow import train
        import json

        print(
            json.dumps(
                train(
                    args.snapshot, args.policy, args.output, args.allow_different_state
                ),
                indent=2,
            )
        )
        return
    plans = {
        "shared": "docs/shared-forecast-plan.json",
        "blend": "docs/shared-blend-plan.json",
        "recency": "docs/shared-recency-plan.json",
    }
    output = create_run(args.output, plans[args.study])
    from src.models.shared_study import run
    import json

    print(json.dumps(run(output, output / "plan.json"), indent=2))


if __name__ == "__main__":
    main()
