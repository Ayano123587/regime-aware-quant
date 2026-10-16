"""Explicit evaluation of a frozen selected checkpoint on observed 2026 data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-directory', required=True, type=Path)
    parser.add_argument('--report-directory', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--observed-holdout', '--acknowledge-observed', dest='acknowledge_observed', action='store_true', required=True)
    args = parser.parse_args()
    from src.models.uncertainty import evaluate_selected
    result = evaluate_selected(args.checkpoint_directory, args.report_directory,
                               args.predictions, acknowledge_observed=args.acknowledge_observed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
