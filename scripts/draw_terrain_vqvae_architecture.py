"""Publication figure for the implemented terrain-conditioned motion codec."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.draw_pfnn_method_figure import Diagram, ROOT, INK, GRAY, LINE, TEAL, BLUE, GOLD, RED, sha


class CodecArchitecture(Diagram):
    def __init__(self, language, assets):
        super().__init__(language, assets)
        self.figure.set_size_inches(18, 12.4)
        self.ax.set_ylim(1240, 0)
        self.labels = []

    def text(self, *args, **kwargs):
        artist = super().text(*args, **kwargs)
        if hasattr(self, 'labels'):
            self.labels.append(artist)
        return artist

    def module(self, x, y, w, h, title, subtitle, color, *, frozen=False):
        fills = {BLUE: '#edf3fb', TEAL: '#eef7f5', GOLD: '#fbf5e9', RED: '#fcf0ed', GRAY: '#f5f6f8'}
        self.box(x, y, w, h, fill=fills[color], edge=color, radius=3, lw=1.3,
                 dash=(0, (4, 3)) if frozen else None)
        self.text(x+w/2, y+h/2-16, title, size=12, bold=True, ha='center', color=color)
        self.text(x+w/2, y+h/2+19, subtitle, size=9.5, ha='center')

    def lattice(self, cx, cy, *, narrow=False):
        # An oblique drawing of the actual 8 x 8 x 8 regular query lattice.
        width, depth, height = (58, 77, 88) if narrow else (73, 55, 88)

        def project(x, y, z):
            return (cx + width*(x-.5) + depth*(z-.5)*.62,
                    cy - height*(y-.5) - depth*(z-.5)*.42)

        faces = [((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
                 ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
                 ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))]
        for face in faces:
            self.ax.add_patch(Polygon([project(*p) for p in face], facecolor='#eef7f5',
                                      edgecolor=TEAL, linewidth=.8, alpha=.8, zorder=4))
        axis = np.linspace(0, 1, 8)
        for y in axis:
            for z in axis:
                row = np.array([project(x, y, z) for x in axis])
                self.ax.scatter(row[:, 0], row[:, 1], s=2.0, color=TEAL, alpha=.26, zorder=5)
        for corner in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            for dim in range(3):
                end = list(corner)
                if end[dim] == 0:
                    end[dim] = 1
                    a, b = project(*corner), project(*end)
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=.8, zorder=6)

    def codebook(self, x, y):
        self.box(x, y, 210, 174, fill='#fbf5e9', edge=GOLD, radius=3, lw=1.3)
        self.text(x+105, y+25, self.tr('多头向量量化', 'Multi-head VQ'), size=12,
                  bold=True, ha='center', color=GOLD)
        for head in range(8):
            for code in range(10):
                selected = code == (head*3+2) % 10
                self.ax.add_patch(Rectangle((x+28+code*16, y+49+head*10), 12, 7,
                    facecolor=GOLD if selected else '#e5d9be', edgecolor='none', zorder=6))
        self.text(x+105, y+144, self.tr('8 组 x 10 个 / 每个 32 维', '8 heads x 10 codes / 32-D'),
                  size=9, ha='center')
        self.text(x+105, y+164, self.tr('最近邻选择 / 码本 EMA 更新', 'Nearest lookup / EMA update'),
                  size=8.5, ha='center', color=GOLD)

    def build(self):
        t = self.tr
        self.text(36, 42, t('地形条件 VQ-VAE：采样、动作编码与条件重建',
                            'Terrain-Conditioned VQ-VAE: Sampling, Tokenization and Reconstruction'),
                  size=20, bold=True)
        self.text(36, 82, t('当前实现 / 动作走离散码本，地形走独立条件分支',
                            'Implemented architecture / Discrete motion codes with a separate terrain branch'),
                  size=11, color=GRAY)
        self.box(1372, 73, 26, 17, edge=BLUE, radius=1, lw=1.2)
        self.text(1410, 82, t('Codec 阶段训练', 'Trainable in codec stage'), size=9)
        self.box(1605, 73, 26, 17, edge=GRAY, radius=1, lw=1.2, dash=(0, (4, 3)))
        self.text(1642, 82, t('冻结', 'Frozen'), size=9)

        self.text(36, 134, t('(a) 以当前状态为中心的地形采样',
                             '(a) Terrain sampling from the current state'), size=13, bold=True)
        self.ax.plot([36, 1764], [155, 155], color=LINE, lw=.8)
        self.text(168, 184, t('配对静态地形', 'Paired static terrain'), size=11, bold=True, ha='center')
        self.terrain(42, 204, 274, 150)
        self.skeleton(174, 299, 72, 3, color=BLUE)
        self.text(168, 359, t('当前 4 帧历史 + 移动指令', '4-frame history + motion command'), size=9.5, ha='center')
        self.line([(320, 265), (368, 265)], color=TEAL, lw=1.5)

        for cx, label, sub, narrow in (
            (443, t('身体周围', 'Body region'), t('水平 +/-0.8 m', 'Horizontal +/-0.8 m'), False),
            (610, t('指令方向', 'Command direction'), t('速度决定前看距离', 'Speed-adaptive lookahead'), True),
            (777, t('实际运动方向', 'Actual travel'), t('由最近动作估计速度', 'Velocity from recent motion'), True),
        ):
            self.lattice(cx, 260, narrow=narrow)
            self.text(cx, 338, label, size=10.5, bold=True, ha='center', color=TEAL)
            self.text(cx, 364, sub, size=8.5, ha='center', color=GRAY)
        self.text(610, 185, t('3 个区域 / 每区域 8 x 8 x 8 点', '3 regions / 8 x 8 x 8 points per region'),
                  size=10.5, ha='center', color=TEAL)

        self.line([(854, 265), (903, 265)], color=TEAL, lw=1.5)
        self.box(903, 210, 222, 114, fill='white', edge=TEAL, radius=3, lw=1)
        self.text(1014, 234, t('1536 点几何观察', '1536-point observation'), size=11, bold=True, ha='center', color=TEAL)
        self.text(1014, 268, t('相对位置 / 距离 / 法线', 'Position / SDF / normal'), size=9.5, ha='center')
        self.text(1014, 300, t('15 维特征 + 有效标记', '15-D features + valid mask'), size=9.5, ha='center')
        self.line([(1125, 265), (1171, 265)], color=TEAL, lw=1.5)
        self.module(1171, 210, 222, 114, t('共享地形编码器', 'Shared terrain encoder'),
                    t('每区域汇总成 8 块', '8 spatial groups per region'), GRAY, frozen=True)
        self.line([(1393, 265), (1441, 265)], color=TEAL, lw=1.5)
        self.box(1441, 210, 307, 114, fill='#eef7f5', edge=TEAL, radius=3, lw=1)
        self.text(1595, 235, t('地形记忆 M', 'Terrain memory M'), size=12, bold=True, ha='center', color=TEAL)
        for i in range(24):
            self.ax.add_patch(Rectangle((1462+(i % 12)*22, 259+(i//12)*19), 17, 13,
                                      facecolor=TEAL, alpha=.35+.4*(i % 3)/2, edgecolor='none', zorder=6))
        self.text(1595, 310, '24 x 128', size=10, ha='center', color=TEAL)

        self.text(1171, 361, t('先在 Root 阶段学习，Codec 阶段冻结',
                               'Learned in Root stage; frozen for codec'), size=9, color=GRAY)
        self.text(36, 413, t('辅助高度分支：附近与前方各 9 x 9 网格，供 Root / Pose 使用。立体采样覆盖 Root 上下各 2 m。',
            'Auxiliary height branch: near + forward 9 x 9 grids for Root / Pose. Volumes span +/-2 m vertically around Root.'),
            size=10, color=GRAY)

        self.text(36, 476, t('(b) 地形条件动作重建 / Codec 训练',
                             '(b) Terrain-conditioned reconstruction / Codec training'), size=13, bold=True)
        self.ax.plot([36, 1764], [497, 497], color=LINE, lw=.8)
        # This single conditioning route terminates at the scene adapter, not at VQ.
        self.line([(1595, 324), (1595, 450), (850, 450), (850, 602)], color=TEAL, lw=1.7)
        self.label(1190, 450, t('地形条件 M', 'Terrain condition M'), color=TEAL, size=10, ha='center')

        self.box(36, 576, 180, 170, fill='white', edge=LINE, radius=3)
        self.text(126, 600, t('真实动作 X', 'Source motion X'), size=11, bold=True, ha='center')
        for i in range(3):
            self.skeleton(77+i*49, 704, 72, i*3, color=BLUE, alpha=.55+i*.2)
        self.text(126, 730, t('32 帧 / 人体特征', '32 frames / body features'), size=9, ha='center', color=GRAY)
        self.line([(216, 660), (264, 660)], color=BLUE, lw=1.6)
        self.module(264, 606, 168, 108, t('动作编码器', 'Motion encoder'), t('时间压缩 4 倍', '4x temporal downsampling'), BLUE)
        self.line([(432, 660), (477, 660)], color=BLUE, lw=1.6)
        self.label(454, 635, '$z_e$', color=BLUE, size=10, ha='center')
        self.codebook(477, 574)
        self.line([(687, 660), (738, 660)], color=GOLD, lw=1.6)
        self.label(713, 635, '$z_q$', color=GOLD, size=10, ha='center')

        self.module(738, 606, 224, 108, t('地形条件适配', 'Terrain adapter'),
                    t('动作特征读取地形记忆', 'Cross-attention to terrain'), TEAL)
        self.line([(962, 660), (1010, 660)], color=GOLD, lw=1.6)
        self.module(1010, 606, 180, 108, t('动作解码器', 'Motion decoder'),
                    t('时间恢复 4 倍', '4x temporal upsampling'), GOLD)
        self.line([(1190, 660), (1240, 660)], color=GOLD, lw=1.6)

        self.box(1240, 576, 211, 170, fill='white', edge=LINE, radius=3)
        self.text(1345, 600, t('重建动作 $\\hat{X}$', 'Reconstruction $\\hat{X}$'), size=11, bold=True, ha='center')
        for i in range(3):
            self.skeleton(1283+i*62, 704, 72, i*3, color=GOLD, alpha=.55+i*.2)
        self.text(1345, 730, t('连续人体动作 / 32 帧', 'Continuous motion / 32 frames'), size=9, ha='center', color=GRAY)
        self.line([(1451, 660), (1500, 660)], color=RED, lw=1.6)

        self.box(1500, 576, 248, 170, fill='#fcf0ed', edge=RED, radius=3, lw=1.3)
        self.text(1624, 600, t('训练目标', 'Training objective'), size=12, bold=True, ha='center', color=RED)
        for yy, label in ((634, t('动作重建  x 1', 'Reconstruction  x 1')),
                          (662, t('关节速度  x 2', 'Joint velocity  x 2')),
                          (690, t('接触 / 脚滑  x 0.01', 'Contact / skate  x 0.01'))):
            self.text(1624, yy, label, size=10, ha='center')
        self.text(1624, 725, t('与源动作逐项比较', 'Compare against source motion'), size=9, ha='center', color=RED)

        self.text(348, 762, t('连续特征：8 x 256', 'Continuous features: 8 x 256'), size=9, ha='center', color=BLUE)
        self.text(582, 785, t('承诺损失 x 0.02', 'Commitment loss x 0.02'), size=10, ha='center', color=GOLD)
        self.text(582, 813, t('直通估计：梯度传回编码器', 'Straight-through encoder gradient'), size=8.7, ha='center', color=GRAY)

        self.box(754, 815, 436, 73, fill='white', edge=LINE, radius=3)
        self.text(972, 837, t('源 Root 轨迹 + 前 4 帧姿态', 'Source Root trajectory + first 4 poses'),
                  size=11.5, bold=True, ha='center')
        self.text(972, 870, t('训练时提供的条件', 'Provided training conditions'), size=9.5, ha='center', color=GRAY)
        self.line([(1100, 815), (1100, 714)], color=BLUE, lw=1.4)
        self.box(771, 755, 158, 44, fill='#edf3fb', edge=BLUE, radius=3, lw=1.3)
        self.text(850, 777, t('高度特征编码', 'Height encoding'), size=10, ha='center', color=BLUE)
        self.line([(850, 815), (850, 799)], color=BLUE, lw=1.4)
        self.line([(850, 755), (850, 714)], color=BLUE, lw=1.4)

        self.line([(126, 746), (126, 928), (1624, 928), (1624, 746)], color=RED, lw=1.1)
        self.label(380, 928, t('真实动作监督 X', 'Source supervision X'), color=RED, size=10, ha='center')
        self.text(754, 966, t('动作编码器 / 解码器与条件适配器学习；码本按 EMA 更新。',
            'Encoder / decoder and adapters learn; codebook vectors use EMA updates.'), size=10, color=GRAY)

        self.text(36, 1020, t('(c) 运行时动作生成',
                               '(c) Runtime motion generation'), size=13, bold=True)
        self.ax.plot([36, 1764], [1041, 1041], color=LINE, lw=.8)
        xs = [36, 390, 744, 1098, 1452]
        labels = [
            (t('历史 / 指令 / 当前地形', 'History / command / terrain'), t('相同观察方式', 'Same observation pipeline'), GRAY),
            ('Root + Pose', t('预测轨迹与动作编号', 'Predict trajectory + code indices'), BLUE),
            (t('码本查表', 'Codebook lookup'), t('使用冻结的动作词表', 'Frozen motion vocabulary'), GOLD),
            (t('同一条件解码器', 'Same conditioned decoder'), t('结合地形 / 历史 / 预测 Root', 'Terrain / context / predicted Root'), TEAL),
            (t('连续人体动作', 'Continuous full-body motion'), t('SOMA 骨骼与蒙皮', 'SOMA skeleton + skinning'), GRAY),
        ]
        for i, (x, (title, subtitle, color)) in enumerate(zip(xs, labels)):
            self.module(x, 1072, 296, 90, title, subtitle, color, frozen=True)
            if i < 4:
                self.line([(x+296, 1117), (xs[i+1], 1117)], color=GRAY, lw=1.4)
        self.line([(538, 1072), (538, 1054), (1246, 1054), (1246, 1072)], color=BLUE, lw=1)
        self.label(936, 1054, t('预测 Root', 'Predicted Root'), color=BLUE, size=8.5, ha='center')
        self.line([(184, 1162), (184, 1181), (1246, 1181), (1246, 1162)], color=TEAL, lw=1)
        self.label(715, 1181, t('当前地形 + 动作历史', 'Current terrain + motion history'), color=TEAL, size=8.5, ha='center')
        self.text(36, 1221, t('图中输入与重建骨架使用同一素材示意结构。有色实线框：Codec 阶段学习（码本 EMA）；虚线框：冻结。',
            'Input / output miniatures reuse the same asset. Colored solid boxes: codec-stage learning (EMA for VQ). Dashed boxes: frozen.'),
            size=8.5, color=GRAY)

    def save_outputs(self, output):
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()
        bounds = self.figure.bbox
        outside = [label.get_text() for label in self.labels
                   if not bounds.contains(*label.get_window_extent(renderer).get_points()[0])
                   or not bounds.contains(*label.get_window_extent(renderer).get_points()[1])]
        if outside:
            raise ValueError(f'Text outside figure: {outside}')
        base = output / f'terrain_conditioned_vqvae_{self.language}'
        files = []
        for extension in ('png', 'svg', 'pdf'):
            path = base.with_suffix('.'+extension)
            self.figure.savefig(path, dpi=260, facecolor='white', metadata={'Creator': 'MotionBricks codec architecture figure'})
            files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(self.figure)
        return files


@torch.inference_mode()
def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = ROOT/'out/pfnn_mixed_20261015/pfnn_map_catalog_v1/003/scene.npz'
    recording = ROOT/'out/pfnn_mixed_20261015/terrain_refresh_diagnosis_v1/forward_walk-24.pt'
    world = torch.load(recording, map_location='cpu', weights_only=False)['world']
    rep = build_native_representation()
    pose = rep.inverse(world, False, return_all=False)
    with np.load(scene, allow_pickle=False) as data:
        assets = dict(height=data['heightfield_y_m'].copy(), origin=data['min_point_world_xyz_m'].copy(),
                      dx=float(data['dx_m']))
    assets.update(world=world.numpy(), joints=pose['posed_joints'][0].numpy(), parents=rep.skeleton.joint_parents.tolist())
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    outputs = []
    for language in ('zh', 'en'):
        diagram = CodecArchitecture(language, assets)
        diagram.build()
        outputs.extend(diagram.save_outputs(output))
    caption = (
        '地形条件动作 VQ-VAE。三组以角色当前状态和移动指令确定的立体网格查询距离、法线及有效性，'
        '经冻结的共享地形编码器得到24个128维条件向量。动作编码器将32帧动作压缩到8个时间位置，'
        '并以8组独立码本（每组10个32维向量）量化。地形特征及源Root高度变化通过条件适配器接入，'
        '随后条件解码器结合Root轨迹和前4帧姿态重建动作。Codec阶段训练编码器、解码器及解码条件适配模块，'
        '码本采用EMA更新。运行时Pose预测动作编号，Root预测轨迹，复用冻结的码本与条件解码链。'
        '输入与输出使用同一素材示意架构，并非新重建精度测量。\n\n'
        'Terrain-conditioned motion VQ-VAE. Three ego-centric volumetric lattices, determined by recent state and '
        'motion commands, query signed distances, normals and validity. A frozen shared encoder produces 24 terrain '
        'tokens of dimension 128. The motion encoder downsamples 32 frames into eight temporal positions and quantizes '
        'them using eight separate codebooks with ten 32-D entries each. A scene adapter attends to terrain and '
        'Root-derived height features; the conditional decoder reconstructs motion using Root trajectories and four '
        'context poses. Codec-stage optimization updates the motion encoder, decoder and decoder-side adapters; '
        'codebooks use EMA. At runtime, Root and Pose predict trajectories and indices and reuse the frozen decoding '
        'chain. Input and output miniatures reuse the same asset for architectural illustration, not reconstruction measurements.\n'
    )
    (output/'caption.txt').write_text(caption, encoding='utf-8')
    sources = [ROOT/'motionbricks/data/unified_scene_field.py', ROOT/'motionbricks/data/unified_terrain.py',
               ROOT/'motionbricks/data/unified_full_training.py',
               ROOT/'motionbricks/vqvae/neural_modules/quantize_cnn_multihead.py']
    receipt = dict(outputs=outputs, language_versions=['zh', 'en'], actual_optimizer_updates=0,
                   codebook_shape=[8, 10, 32], scene_tokens=[24, 128], queries=[3, 8, 8, 8],
                   source_sha256={str(path): sha(path) for path in sources},
                   drawing_source_sha256=sha(Path(__file__)), scene_sha256=sha(scene), recording_sha256=sha(recording),
                   illustrations='Existing static map and recorded poses reused as schematic input/output artwork.')
    (output/'figure_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), files=len(outputs)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/figures/terrain_vqvae_architecture_20261017')
    main(parser.parse_args())
