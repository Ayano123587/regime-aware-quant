"""Aggregate-only candidate report; no market data or test replay."""
import argparse,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args();r=json.loads((args.output/'selection.json').read_text())
    print(json.dumps({'selected':r['selected'],'candidates':[{'candidate':x['spec']['name'],'validation_rmse':x['pooled_validation_rmse']} for x in r['runs']],'failures':r['failures'],'test_used_for_selection':r['test_used_for_selection']},indent=2))


if __name__=='__main__':main()
