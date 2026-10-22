"""Compare actual eager model, live backend and committed packet prefetch locally."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.inference_mode()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((args.artifact/'export.json').read_text())
    for name,digest in receipt['files'].items():
        if sha256(args.artifact/name)!=digest:raise ValueError('Private receipt mismatch:'+name)
    reports=[]
    for entry in receipt['entries']:
        label=entry['entry'];command=entry['command']
        runtime=A100TrialRuntime(args.bundle,receipt['source_bundle_sha256'],'cuda:0',args.artifact/(label+'-scene.npz'),
            'Accepted PARC continuity review','eager',True,args.artifact/'weights.pt',True,True)
        serial=runtime.create_backend('terrain');serial.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
        session=SomaLabSession(runtime,'terrain')
        request=dict(mode='terrain',velocity_xz=command['velocity_xz_mps'],heading=command['heading_rad'],
                     runtime_controls=dict(JOINT_RUNTIME_DEFAULTS))
        source_file=(args.recordings/'recordings/feedback'/('generated_afe4ea1b1d92b874-up-speed0.7-40f.npz'
            if label=='up' else 'official_dc03d6484b176db0-down-speed1.0-64f.npz'))
        with np.load(source_file,allow_pickle=False) as source:reference=torch.from_numpy(source['world'].copy()).cuda()[None]
        parts=[];times=[];max_stream_error=0.
        first=serial.predict_plan(command['velocity_xz_mps'],command['heading_rad'])
        metadata,payload=session.plan(dict(request,request_id=1))
        torch.testing.assert_close(session.pending_playback['motion'],first['planned_motion'],atol=1e-5,rtol=1e-5)
        parts.append(first['planned_motion'].clone())
        for block in range(1,reference.shape[1]//8):
            serial.history=first['published_history'].clone()
            first=serial.predict_plan(command['velocity_xz_mps'],command['heading_rad'])
            actual_before=session.backend.history.clone()
            started=time.perf_counter()
            metadata,payload=session.prefetch(dict(request,type='prefetch',request_id=block+1,
                source_request_id=block,played_frame_count=0))
            times.append((time.perf_counter()-started)*1000)
            candidate=session.pending_playback['prepared']['motion']
            difference=float((candidate-first['planned_motion']).abs().max())
            max_stream_error=max(max_stream_error,difference)
            torch.testing.assert_close(candidate,first['planned_motion'],atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(session.backend.history,actual_before,atol=0,rtol=0)
            assert metadata['history_source']=='immutable_published_model_frames'
            assert metadata['geometry_format']=='soma77-rig-v1'
            packet=np.frombuffer(payload,dtype='<f4').reshape(8,-1)
            np.testing.assert_array_equal(packet[:,:3],candidate[0,:,:3].cpu().numpy())
            session.acknowledge_played(dict(request_id=block,frame_count=8))
            parts.append(candidate.clone())
        session.acknowledge_played(dict(request_id=reference.shape[1]//8,frame_count=8))
        world=torch.cat(parts,1)
        row=dict(entry=label,frames=world.shape[1],source_recording=str(source_file),
            local_stream_vs_serial_max_feature_error=max_stream_error,
            local_vs_a100_max_feature_error=float((world-reference).abs().max()),
            local_vs_a100_root_max_m=float((world[:,:,:3]-reference[:,:,:3]).abs().max()),
            end_to_end_packet_median_ms=float(np.median(times)),end_to_end_packet_p95_ms=float(np.quantile(times,.95)),
            max_packet_ms=max(times),packet_times_ms=times,
            official_rig_vertex_error_m=runtime.terrain_quality['rig_transport_max_vertex_error_m'],
            initial_model_progress=runtime.terrain_quality,postprocessing=None)
        reports.append(row);write_json(args.output/'progress.json',dict(rows=reports));print(json.dumps(row),flush=True)
        del runtime,serial,session,first,candidate,parts,world,reference
        gc.collect();torch.cuda.empty_cache()
    write_json(args.output/'complete.json',dict(rows=reports,browser_wasd_verified=False,model_weights_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','bundle','recordings','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
