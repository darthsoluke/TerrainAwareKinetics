"""Audit all64 candidates and archive failures without admitting bad control labels."""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):value.update(part)
    return value.hexdigest()


def main(args):
    root=args.batch.parent;output=root/'screening_v1';output.mkdir(exist_ok=False)
    complete=json.loads((args.batch/'complete.json').read_text())
    if complete['total_candidates']!=64:raise ValueError('Generation inventory incomplete')
    rows=[];geometry_hashes=[]
    for layout in range(8):
        folder=args.batch/f'layout_{layout:02d}'/'generation'
        report=json.loads((folder/'generation.json').read_text())
        course=json.loads((folder/'course/course_manifest.json').read_text())
        geometry_hashes.append(course['terrain_sha256'])
        for row in report['candidates']:
            if digest(Path(row['motion']))!=row['sha256'] or digest(Path(row['intent']))!=row['intent_sha256']:
                raise ValueError('Raw motion or commands changed')
            failures=[]
            if not row['geometric_route_covered']:failures.append('complete_up_and_down_route_not_verified')
            stops=row['stop_response']['stops']
            if len(stops)!=2:failures.append('two_requested_stops_not_both_present')
            for stop in stops:
                prefix=stop['phase']+':'
                if stop['final_half_second_speed_p90_mps']>.15:failures.append(prefix+'not_stationary')
                if stop['final_half_second_drift_m']>.08:failures.append(prefix+'drift')
                if not stop['stopped_inside_flight']:failures.append(prefix+'not_mid_flight')
                if stop['resume_progress_m']<.25:failures.append(prefix+'resume_not_verified')
            rows.append(dict(layout=layout,candidate_id=row['candidate_id'],split=report['split'],
                frames=row['frames'],duration_s=row['duration_s'],complete_route=row['geometric_route_covered'],
                stops=stops,control_prescreen_pass=not failures,rejection_reasons=failures,
                motion_sha256=row['sha256'],intent_sha256=row['intent_sha256']))
    if len(rows)!=64 or len(set(geometry_hashes))!=8:raise ValueError('Geometry/candidate split inventory mismatch')
    stop_rows=[s for r in rows for s in r['stops']]
    eligible=sum(r['control_prescreen_pass'] for r in rows)
    result=dict(total_candidates=64,complete_routes=sum(r['complete_route'] for r in rows),
        recorded_stop_intervals=len(stop_rows),single_stop_and_resume_pass=sum(s['kinematic_stop_response'] for s in stop_rows),
        stationary_tail_intervals=sum(s['final_half_second_speed_p90_mps']<=.15 and s['final_half_second_drift_m']<=.08 for s in stop_rows),
        control_eligible_candidates=eligible,training_admitted=0,training_started=False,
        physics_soma_status='not_run_no_control_eligible_candidates' if not eligible else 'pending',
        all_failures_preserved=True,scene_split_hashes=geometry_hashes,rows=rows,
        scope='Native kinematic command-response gate only. Passing is not physics/SOMA approval; failed stop targets are not training labels.')
    (output/'complete.json').write_text(json.dumps(result,indent=2))
    archive=root/'stair64_candidates_v1.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for name in ('batch_v1','code_v1','screening_v1'):
            stream.add(root/name,arcname=name)
    receipt=dict(archive=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,
        total_candidates=64,training_admitted=0,approved_orange_unchanged=True)
    receipt_path=root/'stair64_candidates_v1.receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2))
    config=configparser.ConfigParser(interpolation=None);config.read(args.drive_config)
    if config['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Not the user-owned Drive OAuth client')
    env=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    for path in (archive,receipt_path,output/'complete.json'):
        subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(path),
            'private-store:motionbricks/parc_stair64_20261013/'+path.name,'--immutable','--checksum',
            '--transfers','1','--checkers','1','--tpslimit','1','--retries','2'],env=env,check=True)
    (output/'upload-complete.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','scene_split_hashes')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--drive-config',type=Path,required=True)
    main(parser.parse_args())
