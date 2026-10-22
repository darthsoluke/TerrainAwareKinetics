"""Finish the active worker, then retry its failed scenes with the corrected source."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1),required=True)
    parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args()
    while True:
        cmdline=Path(f'/proc/{args.pid}/cmdline')
        if not cmdline.exists():
            break
        text=cmdline.read_bytes().decode().replace('\0',' ')
        if 'run_parc_diverse_collection.py' not in text or str(args.output) not in text:
            break
        time.sleep(30)
    result=subprocess.run([sys.executable,'-u',str(args.work/'scripts/run_parc_diverse_collection.py'),
        '--output',str(args.output),'--work',str(args.work),'--gpu',str(args.gpu),'--retry-failed'])
    if result.returncode:
        raise RuntimeError('Recovery pass failed')
    (args.output/f'recovery_{args.gpu}.json').write_text(json.dumps({'completed':True,'exit_code':0}))


if __name__=='__main__':
    main()
