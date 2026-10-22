"""Bind the unchanged final model to an original PFNN demo heightmap."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
import zlib

import numpy as np

from scripts.inventory_pfnn_source import OFFICIAL_URL, RangeReader


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def convert_heightmap(raw, horizontal_cm, vertical_cm, spawn_xz):
    if raw.ndim != 2 or min(raw.shape) < 2 or not np.isfinite(raw).all():
        raise ValueError('A complete finite original heightmap is required')
    dx = horizontal_cm*.01
    origin = np.array([-(raw.shape[0]//2)*dx, 0., -(raw.shape[1]//2)*dx], np.float32)
    grid = (np.asarray(spawn_xz)-origin[[0, 2]])/dx
    low = np.floor(grid).astype(int)
    fraction = grid-low
    if np.any(low < 0) or np.any(low+1 >= raw.shape):
        raise ValueError('PFNN spawn is outside the source map')
    x, z = low
    a, b = fraction
    spawn_height = ((1-a)*(1-b)*raw[x, z]+a*(1-b)*raw[x+1, z]
                    +(1-a)*b*raw[x, z+1]+a*b*raw[x+1, z+1])
    # PFNN stores data[x][z]; the SOMA scene contract stores heights[z, x].
    height = ((raw.astype(np.float64)-spawn_height)*vertical_cm*.01).T.astype(np.float32)
    return height, origin, np.float32(dx), float(spawn_height)


def display_mesh(height, origin, dx, ao):
    zs = np.unique(np.r_[np.arange(0, height.shape[0], 2), height.shape[0]-1])
    xs = np.unique(np.r_[np.arange(0, height.shape[1], 2), height.shape[1]-1])
    z, x = np.meshgrid(zs, xs, indexing='ij')
    vertices = np.stack((origin[0]+x*dx, height[z, x], origin[2]+z*dx), -1).astype(np.float32)
    ids = np.arange(len(zs)*len(xs), dtype=np.uint32).reshape(len(zs), len(xs))
    a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:], ids[1:, 1:]
    faces = np.stack((np.stack((a,b,c),-1),np.stack((c,b,d),-1)), -2).reshape(-1,3)
    ambient = np.clip(ao.T[z, x], 0, 1)
    colors = ((.6+.4*ambient[...,None])*np.array([.64,.69,.65])).astype(np.float32)
    return dict(vertices=vertices.reshape(-1,3), faces=faces, colors=colors.reshape(-1,3))


def package(args, reader=None, source=None):
    args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.parent/'acceptance.json').read_text())
    if sha(args.parent/parent['checkpoint_file']) != parent['checkpoint_sha256']:
        raise ValueError('Final model artifact changed')
    reader = reader or RangeReader(OFFICIAL_URL)
    source = source or zipfile.ZipFile(reader)
    roots = [name[:-len('demo/pfnn.cpp')] for name in source.namelist() if name.endswith('/demo/pfnn.cpp')]
    if len(roots) != 1:
        raise ValueError('Ambiguous official PFNN demo')
    prefix = roots[0]
    verified = {}
    map_folder = 'demo/heightmaps' if (args.source/'demo/heightmaps'/(args.map+'.txt')).exists() else 'data/heightmaps'
    assets = ['demo/pfnn.cpp', map_folder+'/'+args.map+'.txt']
    if (args.source/map_folder/(args.map+'_ao.txt')).exists():
        assets.append(map_folder+'/'+args.map+'_ao.txt')
    for relative in assets:
        item = source.getinfo(prefix+relative)
        local = args.source/relative
        data = local.read_bytes()
        if len(data) != item.file_size or zlib.crc32(data)&0xffffffff != item.CRC:
            data = data.replace(b'\r\n', b'\n')
        if len(data) != item.file_size or zlib.crc32(data)&0xffffffff != item.CRC:
            data = source.read(item)
        if len(data) != item.file_size or zlib.crc32(data)&0xffffffff != item.CRC:
            raise ValueError('Official demo asset verification failed')
        target = args.output/Path(relative).name
        target.write_bytes(data)
        verified[relative] = dict(sha256=sha(target), official_crc32=item.CRC, bytes=item.file_size)
    code = (args.output/'pfnn.cpp').read_text()
    horizontal = float(re.search(r': hscale\(([0-9.]+)\)', code).group(1))
    vertical = float(re.search(r'^\s*, vscale\(([0-9.]+)\)', code, re.M).group(1))
    raw = np.loadtxt(args.output/(args.map+'.txt'), dtype=np.float32)
    body = next((body for body in re.findall(r'static void load_world\d\(void\) \{(.*?)\n\}',code,re.S)
                 if args.map+'.txt' in body), None)
    if body is not None:
        spawn = [float(v)*.01 for v in re.search(r'reset\(glm::vec2\(([-0-9.]+), ([-0-9.]+)\)\)', body).groups()]
        spawn_policy = 'original PFNN demo reset'
    else:
        from scipy.ndimage import maximum_filter, minimum_filter
        radius = int(np.ceil(.55/(horizontal*.01)))
        relief = maximum_filter(raw,2*radius+1)-minimum_filter(raw,2*radius+1)
        margin = int(np.ceil(3.5/(horizontal*.01)))
        x,z = np.indices(raw.shape)
        score = relief.astype(np.float64)+1e-6*((x-raw.shape[0]//2)**2+(z-raw.shape[1]//2)**2)
        score[:margin]=np.inf; score[-margin:]=np.inf
        score[:,:margin]=np.inf; score[:,-margin:]=np.inf
        ix,iz = np.unravel_index(score.argmin(),score.shape)
        spawn = [(ix-raw.shape[0]//2)*horizontal*.01,(iz-raw.shape[1]//2)*horizontal*.01]
        spawn_policy = 'fixed low-relief entry chosen from static terrain before inference'
    ao_path = args.output/(args.map+'_ao.txt')
    ao = np.loadtxt(ao_path,dtype=np.float32) if ao_path.exists() else np.ones_like(raw)
    if ao.shape[0] != raw.shape[0] or ao.shape[1] not in (raw.shape[1],raw.shape[1]+1):
        raise ValueError('Original terrain ambient-occlusion array shape differs')
    ao = ao[:, :raw.shape[1]]
    height, origin, dx, vertical_reference = convert_heightmap(raw, horizontal, vertical, spawn)
    with np.load(args.parent/parent['scene_file'], allow_pickle=False) as original:
        history = original['initial_history'].copy()
        seed_metadata = json.loads(str(original['metadata_json']))
    translation = np.asarray(spawn, np.float32)-history[0,-1,[0,2]]
    history[..., [0,2]] += translation
    mesh = args.output/'terrain_mesh.npz'
    np.savez_compressed(mesh, **display_mesh(height, origin, float(dx), ao))
    label = 'PFNN Urban' if 'urban' in args.map else 'PFNN '+args.map.split('_')[1]
    metadata = dict(schema='motionbricks.full_corpus_scene.v1', bundle_sha256=parent['bundle_sha256'],
        history_frames=4, future_motion_included=False, coordinate_frame='soma_world_y_up',
        provenance=dict(source='PFNN original heightmap '+args.map, official_url=OFFICIAL_URL,
            assets=verified, source_shape_xz=list(raw.shape), horizontal_cm=horizontal, vertical_cm=vertical,
            world_units='metres, converted from author runtime centimetres',
            vertical_reference_raw=vertical_reference, terrain_rebase='one fixed vertical translation at the declared spawn',
            spawn_policy=spawn_policy,
            source_seed=seed_metadata, source_seed_xz_translation_m=translation.tolist(),
            author_collision_forces_and_ik_enabled=False, original_model_weights_unchanged=True),
        display_mesh=dict(file=mesh.name, sha256=sha(mesh), stride=2,
            semantics='Original PFNN terrain surface; author default stride2 display, full-resolution model queries'),
        world=dict(label=label, size_m=[height.shape[1]*float(dx),height.shape[0]*float(dx)],
            spawns=[dict(id='hub', label=label, spawn_xz=spawn, heading=0.)], default_spawn='hub'))
    scene_path = args.output/'scene.npz'
    np.savez_compressed(scene_path, heightfield_y_m=height, min_point_world_xyz_m=origin,
        dx_m=dx, initial_history=history, metadata_json=json.dumps(metadata))
    os.link(args.parent/parent['checkpoint_file'], args.output/'weights.pt')
    receipt = dict(parent, checkpoint_file='weights.pt', scene_file='scene.npz', scene_sha256=sha(scene_path))
    (args.output/'acceptance.json').write_text(json.dumps(receipt, indent=2))
    report = dict(packaged=True, original_weights_sha256=parent['checkpoint_sha256'],
        official_assets=verified, network_bytes=reader.downloaded,
        full_resolution_shape_zx=list(height.shape), dx_m=float(dx), size_m=metadata['world']['size_m'],
        height_range_m=[float(height.min()),float(height.max())], spawn_xz=spawn,
        scene_sha256=receipt['scene_sha256'], metadata=metadata,
        actual_optimizer_updates=0, quality_approved=False, delivery_approved=False)
    (args.output/'map_receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','official_assets')}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--map', default='hmap_urban_001_smooth')
    package(p.parse_args())
