"""Compare matched held-out model rollouts, including regressions and failed tasks."""

import argparse
import json
from pathlib import Path

from motionbricks.data.terrain_quality_selection import pilot_decision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initial','native','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads(path.read_text()) for path in (args.initial,args.native,args.candidate)]
    if any('absolute_gates_passed' not in report for report in reports):
        raise ValueError('A pilot evaluation is still incomplete')
    for section,key in (('short','source_id'),('flat','source_id'),('walls','source_id')):
        identities=[[row[key] for row in report[section]] for report in reports]
        if any(identity!=identities[0] for identity in identities):
            raise ValueError('Pilot reports assess different held-out cases')
    result=pilot_decision(*reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
