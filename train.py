"""Validation-only chronological training; never silently replays observed test."""
import argparse
from pathlib import Path


def create_run(output,plan):
    output=Path(output)
    if output.exists():raise ValueError('Output already exists; preserve historical artifacts and choose a fresh directory')
    content=Path(plan).read_bytes();output.mkdir(parents=True);(output/'plan.json').write_bytes(content)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study',choices=['selected','shared','blend','recency']);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.study == 'selected':
        from src.models.uncertainty import train_selected
        train_selected(args.output)
        return
    plans={'shared':'docs/shared-forecast-plan.json','blend':'docs/shared-blend-plan.json','recency':'docs/shared-recency-plan.json'}
    output=create_run(args.output,plans[args.study])
    from src.models.shared_study import run
    import json
    print(json.dumps(run(output,output/'plan.json'),indent=2))


if __name__=='__main__':main()
