"""Official SOMA source/A/B on paired terrain, with identical fixed cameras."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
import imageio_ffmpeg
import torch

from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.data.soma_skin_reference import load_official_skinner


LABELS = ('PARC source', 'A: absolute height', 'B: relative height')
COLORS = ('#747b80', '#ce772f', '#258c7c')


def terrain_faces(hf, origin, dx, low, high):
    """Visible box tops and risers, with no invented plane under unknown space."""
    polygons, colors = [], []
    bottom = low[1] - .3
    for z in range(hf.shape[0]):
        cz = float(origin[2] + z * dx)
        if not low[2] - dx <= cz <= high[2] + dx:
            continue
        for x in range(hf.shape[1]):
            cx = float(origin[0] + x * dx)
            if not low[0] - dx <= cx <= high[0] + dx:
                continue
            top = float(hf[z, x])
            if top < bottom:
                continue
            x0, x1, z0, z1 = cx-dx/2, cx+dx/2, cz-dx/2, cz+dx/2
            polygons.append([(x0,z0,top), (x1,z0,top), (x1,z1,top), (x0,z1,top)])
            colors.append('#d2d7d8' if (x+z) % 2 else '#e0e3e4')
            sides = [(-1,0,[(x0,z0),(x0,z1)]), (1,0,[(x1,z1),(x1,z0)]),
                     (0,-1,[(x1,z0),(x0,z0)]), (0,1,[(x0,z1),(x1,z1)])]
            for ox, oz, edge in sides:
                neighbor = float(hf[z+oz,x+ox]) if 0 <= z+oz < hf.shape[0] and 0 <= x+ox < hf.shape[1] else bottom
                lower = max(neighbor, bottom)
                if lower >= top - 1e-6:
                    continue
                polygons.append([(*edge[0],top), (*edge[1],top), (*edge[1],lower), (*edge[0],lower)])
                colors.append('#a6afb2')
    return polygons, colors


@torch.inference_mode()
def render(first, second, output, skinner, window, label):
    name = f'window-{window:02d}-played_history.npz'
    with np.load(first / name, allow_pickle=False) as data:
        a = {key: data[key].copy() for key in data.files}
    with np.load(second / name, allow_pickle=False) as data:
        b = {key: data[key].copy() for key in data.files}
    for key in ('source', 'roots_source', 'local_source', 'hf', 'min_point', 'dx', 'fps'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Matched visualization sources differ: ' + key)
    motions = ((a['roots_source'], a['local_source']),
               (a['roots_prediction'], a['local_prediction']),
               (b['roots_prediction'], b['local_prediction']))
    meshes = []
    for roots, rotations in motions:
        positions = torch.from_numpy(roots).float()[:, None].expand(-1, 30, -1)
        vertices = skinner.skin(torch.from_numpy(rotations).float(), positions, rot_is_global=False)
        if not torch.isfinite(vertices).all():
            raise ValueError('Nonfinite official SOMA skinning')
        meshes.append(vertices.numpy())
    points = np.concatenate(meshes)
    world_low, world_high = points.min((0, 1))-.5, points.max((0, 1))+.5
    camera_origin = motions[0][0].copy()
    camera_origin[:, 1] = 0
    relative_points = np.concatenate([mesh-camera_origin[:,None] for mesh in meshes])
    low, high = relative_points.min((0, 1)), relative_points.max((0, 1))
    low -= [.32, .25, .32]
    high += [.32, .18, .32]
    span = np.maximum(high-low, [1.6, 1.8, 1.6])
    center = (low+high)/2
    low, high = center-span/2, center+span/2
    polygons, terrain_colors = terrain_faces(a['hf'], a['min_point'], float(a['dx']), world_low, world_high)
    quads = np.asarray(polygons, dtype=np.float32).reshape(-1,4,3)
    terrain_triangles = quads[:, [[0,1,2],[0,2,3]]].reshape(-1,3,3)
    terrain_rgb = np.repeat([matplotlib.colors.to_rgb(c) for c in terrain_colors], 2, axis=0)
    faces = skinner.faces.numpy()
    fig = plt.figure(figsize=(13.5, 5.8), dpi=100, facecolor='#f8fafb')
    artists = []
    light = np.array([.4, -.7, 1.])
    light /= np.linalg.norm(light)
    for panel, (mesh, title, color) in enumerate(zip(meshes, LABELS, COLORS), 1):
        ax = fig.add_subplot(1, 3, panel, projection='3d')
        ax.set_facecolor('#f8fafb')
        ax.set_proj_type('ortho')
        ax.set(xlim=(low[0],high[0]), ylim=(low[2],high[2]), zlim=(low[1],high[1]),
               xlabel='X (m)', ylabel='Z (m)', zlabel='Y (m)')
        ax.set_title(title, color=color, fontsize=14, fontweight='bold', pad=12)
        ax.set_box_aspect(span[[0,2,1]])
        ax.view_init(13, -57)
        ax.tick_params(labelsize=7, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['color'] = (.65, .7, .73, .15)
        # One collection lets the renderer depth-sort body and terrain triangles
        # together; separate collection-level sorting can falsely hide limbs.
        surface = Poly3DCollection([], linewidths=0, edgecolors='none', zsort='average')
        ax.add_collection3d(surface)
        roots = motions[panel-1][0]
        trail, = ax.plot([], [], [], color=color, linewidth=1, alpha=.45)
        artists.append((surface, mesh, np.array(matplotlib.colors.to_rgb(color)), ax, trail, roots))
    fig.suptitle(label + ' | continuous generation on the same terrain', fontsize=14, y=.98)
    timestamp = fig.text(.5, .035, '', ha='center', fontsize=10, color='#3f4b50')
    fig.text(.5, .009, 'Official SOMA skin | cameras follow the same source XZ | no IK, foot locking or motion smoothing',
             ha='center', fontsize=9, color='#657177')
    fig.subplots_adjust(left=.015, right=.97, bottom=.12, top=.86, wspace=.06)
    def update(frame):
        for surface, mesh, color, ax, trail, roots in artists:
            origin = camera_origin[frame]
            ax.set_xlim(low[0]+origin[0], high[0]+origin[0])
            ax.set_ylim(low[2]+origin[2], high[2]+origin[2])
            triangles = mesh[frame][:, [0,2,1]][faces]
            normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-10)
            intensity = .46 + .54*np.maximum(normals@light, 0)
            surface.set_verts(np.concatenate((terrain_triangles, triangles)))
            surface.set_facecolor(np.concatenate((terrain_rgb, intensity[:,None]*color)))
            past = roots[max(0,frame-7):frame+1]
            trail.set_data_3d(past[:,0], past[:,2], past[:,1])
        timestamp.set_text(f'{frame/30:.2f} s    |    frame {frame+1}/{len(meshes[0])}    |    evaluation, not quality approved')
    output.mkdir(parents=True, exist_ok=True)
    stem = f'comparison-{window:02d}'
    video = output / (stem + '.mp4')
    writer = imageio_ffmpeg.write_frames(str(video), (1350,580), fps=30,
        codec='libx264', quality=8, macro_block_size=2,
        output_params=['-movflags', '+faststart', '-threads', '2'])
    writer.send(None)
    frames = []
    try:
        for frame in range(len(meshes[0])):
            update(frame)
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            writer.send(pixels)
            image = Image.fromarray(pixels)
            frames.append(image.resize((1080,464), Image.Resampling.LANCZOS))
            if frame == min(24, len(meshes[0])-1):
                image.save(output / (stem + '.png'))
    finally:
        writer.close()
        plt.close(fig)
    times = np.rint(np.arange(len(frames)+1)*1000/30/10).astype(int)*10
    frames[0].save(output / (stem + '.gif'), save_all=True, append_images=frames[1:],
                   duration=np.diff(times).tolist(), loop=0, disposal=2)
    row = {'window': window, 'label': label, 'frame_count': len(frames), 'fps': 30,
           'video': video.name, 'gif': stem+'.gif', 'preview': stem+'.png',
           'source_sha256_A': hashlib.sha256((first/name).read_bytes()).hexdigest(),
           'source_sha256_B': hashlib.sha256((second/name).read_bytes()).hexdigest(),
           'low_terrain_geometry_clipped_below_y_m': float(world_low[1]-.3),
           'camera': 'all panels follow source XZ only; same fixed height, orientation and scale',
           'postprocessing': None, 'skin': 'official SOMASkin, all vertices and influences'}
    print(json.dumps(row), flush=True)
    return row


def gallery(output, rows):
    sections = ''.join(f'<section><h2>{html.escape(r["label"])}</h2>'
        f'<video controls loop muted playsinline preload="metadata" poster="{r["preview"]}" src="{r["video"]}"></video></section>'
        for r in rows)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MotionBricks A/B</title><style>body{margin:0;background:#f8fafb;color:#26373c;font:15px system-ui,sans-serif}
main{max-width:1450px;margin:0 auto;padding:24px}h1{font-size:24px;margin:0 0 12px}p{line-height:1.6;margin:8px 0}
.legend{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}.source{color:#747b80}.a{color:#aa5917}.b{color:#187b6b}
label{display:inline-flex;gap:8px;align-items:center;margin:12px 0}select{font:inherit;padding:4px}section{border-top:1px solid #dce3e5;margin-top:22px;padding-top:12px}
h2{font-size:17px}video{display:block;width:100%;height:auto}a{color:#187b6b}</style><main>
<h1>MotionBricks A/B</h1><p>同一地形、同一平面目标，连续回读生成动作。每段 32 帧，原始速度约 1.07 秒。</p>
<div class="legend"><strong class="source">左：PARC 原始动作</strong><strong class="a">中：A 绝对高度组</strong><strong class="b">右：B 相对高度组</strong></div>
<p>两组均完成 Tokenizer、Root、Pose 各 2,000 步。相对高度组尚未通过闭环质量验收，视频保留漂浮、穿地和滑步。</p>
<label>播放速度 <select id="rate"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label>'''
    page += sections + '''<script>const videos=[...document.querySelectorAll('video')];const rate=document.querySelector('#rate');
function apply(){videos.forEach(v=>v.playbackRate=Number(rate.value))}rate.onchange=apply;videos.forEach(v=>v.addEventListener('loadedmetadata',apply));apply();</script></main></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('absolute', 'observed', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--windows', default='0,1,2')
    args = p.parse_args()
    torch.set_num_threads(4)
    skinner = load_official_skinner(SOMASkeleton30())
    labels = ('Mixed terrain', 'Platform', 'Climbing', 'Boxes', 'Mixed motion', 'Mixed terrain 2', 'Platform 2', 'Climbing 2')
    rows = [render(args.absolute, args.observed, args.output, skinner, i, labels[i])
            for i in map(int,args.windows.split(','))]
    gallery(args.output, rows)
    (args.output/'manifest.json').write_text(json.dumps({'rows': rows, 'quality_approved': False,
        'scope': 'offline held-out continuous generation; identical source and scene across A/B'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
