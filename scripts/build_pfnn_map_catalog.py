"""Package all unique original PFNN heightmaps without changing model weights."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

from scripts.inventory_pfnn_source import OFFICIAL_URL, RangeReader
from scripts.package_pfnn_demo_map import package, sha


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    reader=RangeReader(OFFICIAL_URL)
    source=zipfile.ZipFile(reader)
    names=sorted({path.stem for folder in ('data/heightmaps','demo/heightmaps')
                  for path in (args.source/folder).glob('*.txt') if not path.stem.endswith('_ao')})
    entries=[]
    for name in names:
        map_id='urban' if 'urban' in name else name.split('_')[1]
        folder=args.output/map_id
        if map_id=='urban':
            folder.mkdir()
            for path in args.urban.iterdir():
                if path.is_file():os.link(path,folder/path.name)
            report=json.loads((folder/'map_receipt.json').read_text())
        else:
            report=package(SimpleNamespace(output=folder,source=args.source,parent=args.parent,map=name),reader,source)
        entry=dict(id=map_id,label='PFNN Urban' if map_id=='urban' else 'PFNN '+map_id,
            artifact=map_id,source_map=name,category='Demo' if (args.source/'demo/heightmaps'/(name+'.txt')).exists() else 'Terrain',
            scene_sha256=report['scene_sha256'],size_m=report['size_m'])
        entries.append(entry)
        (args.output/'progress.json').write_text(json.dumps(dict(completed_maps=entries,total_maps=len(names)),indent=2))
    catalog=dict(schema='motionbricks.pfnn_map_catalog.v1',default_map='urban',entries=entries,
        unique_map_count=len(entries),all_source_maps=names,actual_optimizer_updates=0,
        checkpoint_sha256=json.loads((args.parent/'acceptance.json').read_text())['checkpoint_sha256'],
        training_data_modified=False,quality_approved=False,delivery_approved=False)
    (args.output/'catalog.json').write_text(json.dumps(catalog,indent=2))
    print(json.dumps(dict(completed=True,maps=len(entries),catalog=str(args.output/'catalog.json'),
        catalog_sha256=sha(args.output/'catalog.json'),network_bytes=reader.downloaded)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','urban','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
