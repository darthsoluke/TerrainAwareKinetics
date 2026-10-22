"""Create a loopback-only task proxy from an already configured private node."""
import argparse
from pathlib import Path

import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--node',default='HKG A')
    parser.add_argument('--port',type=int,default=27891)
    args=parser.parse_args()
    source=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    candidates=[node for node in source.get('proxies',[]) if node.get('name')==args.node]
    if len(candidates)!=1:raise ValueError('Requested existing node must be unique')
    value={'mixed-port':args.port,'allow-lan':False,'bind-address':'127.0.0.1',
        'mode':'rule','log-level':'warning','ipv6':False,'proxies':candidates,
        'rules':['MATCH,'+args.node]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:yaml.safe_dump(value,stream,allow_unicode=True)
    print('Isolated download proxy prepared; system proxy unchanged')


if __name__=='__main__':main()
