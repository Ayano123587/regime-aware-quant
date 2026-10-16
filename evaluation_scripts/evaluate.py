"""Explicit retrospective evaluation after frozen chronological selection."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--observed-holdout',action='store_true',required=True,help='Acknowledge the already-observed retrospective period');args=parser.parse_args()
    selection=json.loads((args.output/'selection.json').read_text())
    if selection['selected']['name']=='v2_full':
        print(json.dumps({'test_evaluated':False,'reason':'Existing v2 retained; no new test scoring'}));return
    from src.models.shared_study import release
    print(json.dumps(release(args.output,args.output/'plan.json'),indent=2))


if __name__=='__main__':main()
