"""Publication-style diagram of the implemented terrain-aware MotionBricks chain."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation


ROOT=Path(__file__).resolve().parents[1]
W,H=1800,1180
INK='#253346'
GRAY='#667486'
LINE='#cdd5df'
TEAL='#237f7b'
GREEN='#4c8b69'
BLUE='#376fb0'
PURPLE='#80669d'
GOLD='#b47d31'
RED='#bd6556'
FONT=Path('C:/Windows/Fonts')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


class Diagram:
    def __init__(self,language,assets):
        self.language=language
        self.assets=assets
        self.figure,self.ax=plt.subplots(figsize=(18,11.8),dpi=220)
        self.figure.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax.set(xlim=(0,W),ylim=(H,0),aspect='equal')
        self.ax.axis('off')
        self.figure.patch.set_facecolor('white')
        self.regular=FontProperties(fname=str(FONT/('msyh.ttc' if language=='zh' else 'arial.ttf')))
        self.bold=FontProperties(fname=str(FONT/('msyhbd.ttc' if language=='zh' else 'arialbd.ttf')))

    def tr(self,zh,en):return zh if self.language=='zh' else en

    def text(self,x,y,text,size=11,color=INK,bold=False,ha='left',va='center',**kwargs):
        return self.ax.text(x,y,text,fontproperties=self.bold if bold else self.regular,
            fontsize=size,color=color,ha=ha,va=va,linespacing=1.45,zorder=10,**kwargs)

    def box(self,x,y,w,h,fill='white',edge=LINE,radius=8,lw=1.0,dash=None):
        patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
            facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=dash or '-',zorder=2)
        self.ax.add_patch(patch)
        return patch

    def line(self,points,color=INK,lw=1.25,dash=False,arrow=True,zorder=5):
        x,y=zip(*points)
        self.ax.plot(x,y,color=color,lw=lw,linestyle=(0,(4,3)) if dash else '-',
            solid_capstyle='round',zorder=zorder)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                mutation_scale=12,color=color,lw=lw,zorder=zorder))

    def label(self,x,y,text,color=GRAY,size=9,ha='left'):
        artist=self.text(x,y,text,size,color,ha=ha)
        artist.set_path_effects([patheffects.withStroke(linewidth=4,foreground='white')])
        artist.set_bbox(dict(facecolor='white',edgecolor='none',pad=2))

    def dot(self,x,y,color):self.ax.scatter([x],[y],s=12,color=color,zorder=8)

    def skeleton(self,cx,base,height,index,color=GRAY,alpha=1):
        joints=self.assets['joints'][index].copy()
        joints-=joints[0]
        projected=.83*joints[:,0]+.55*joints[:,2]
        vertical=joints[:,1]
        scale=height/max(float(vertical.max()-vertical.min()),1e-6)
        x=cx+projected*scale
        y=base-(vertical-vertical.min())*scale
        skip={7,8,9,14,15,20,21}
        for child,parent in enumerate(self.assets['parents']):
            if parent>=0 and child not in skip:
                self.ax.plot(x[[parent,child]],y[[parent,child]],color=color,lw=1.6,alpha=alpha,
                    solid_capstyle='round',zorder=7)
        self.ax.scatter(x[[0,3,11,12,17,18,23,24,27,28]],
            y[[0,3,11,12,17,18,23,24,27,28]],s=6,color=color,alpha=alpha,zorder=8)
        self.ax.add_patch(plt.Circle((x[6],y[6]+4),7,facecolor='white',edgecolor=color,lw=1.4,zorder=8))

    def token_grid(self,x,y,cols,rows,cell=10,colors=None):
        colors=colors or ['#3b8d89','#7cabca','#d1b46f','#987baa','#7e9f7e','#ce9480','#7298a6','#a2ac74']
        for row in range(rows):
            for col in range(cols):
                self.ax.add_patch(Rectangle((x+col*(cell+3),y+row*(cell+2)),cell,cell,
                    facecolor=colors[(row+2*col)%len(colors)],edgecolor='white',lw=.4,zorder=6))

    def terrain(self,x,y,w,h):
        height=self.assets['height']
        origin=self.assets['origin']
        root=self.assets['world'][0,3,:3]
        cell=np.rint((root[[0,2]]-origin[[0,2]])/self.assets['dx']).astype(int)
        radius=85
        x0,x1=max(0,cell[0]-radius),min(height.shape[1],cell[0]+radius)
        z0,z1=max(0,cell[1]-radius),min(height.shape[0],cell[1]+radius)
        patch=height[z0:z1:5,x0:x1:5]
        low,high=float(patch.min()),float(patch.max())
        normalized=(patch-low)/max(high-low,.001)
        # Oblique terrain tiles use actual packaged heights; this is diagram artwork.
        for iz in range(patch.shape[0]-1):
            for ix in range(patch.shape[1]-1):
                def point(i,j):
                    return [x+w*(.12+.67*i/(patch.shape[1]-1)+.23*j/(patch.shape[0]-1)),
                        y+h*(.5+.40*j/(patch.shape[0]-1)-.32*i/(patch.shape[1]-1)-.38*normalized[j,i])]
                shade=.38+.50*normalized[iz,ix]
                color=(.48+.32*shade,.58+.28*shade,.56+.28*shade)
                self.ax.add_patch(Polygon([point(ix,iz),point(ix+1,iz),point(ix+1,iz+1),point(ix,iz+1)],
                    closed=True,facecolor=color,edgecolor='#c5d2cc',linewidth=.12,zorder=4))

    def build(self):
        t=self.tr
        self.text(34,35,t('地形感知 MotionBricks：当前实现的方法框架',
            'Terrain-Aware MotionBricks: Implemented Method'),size=19,bold=True)
        self.text(34,68,t('因果地形观察 · 分阶段学习 · 预测式动作生成 · 播放状态反馈',
            'Causal terrain observations  /  staged learning  /  predictive motion generation  /  playback feedback'),
            size=10,color=GRAY)
        self.text(1765,68,t('SOMA30 / 30 Hz','SOMA30 / 30 Hz'),size=10,color=GRAY,ha='right')

        # (a) Inputs and independent heightfield / volumetric observation paths.
        self.box(30,103,1740,239,fill='#fafcfc',edge='#e0e5eb',radius=9)
        self.text(46,122,t('(a) 当前状态与地形观察','(a) Current state and causal terrain observations'),size=12,bold=True)
        self.box(52,149,330,166)
        self.text(67,170,t('已播放的四帧历史','Four played context frames'),size=11,bold=True)
        for i in range(4):
            self.skeleton(95+i*79,280,76,i,color=BLUE,alpha=.45+.18*i)
            self.text(95+i*79,299,f'$x_{{t-{3-i}}}$' if i<3 else '$x_t$',size=10,ha='center')
        self.box(420,149,175,166)
        self.text(507,170,t('移动意图','User intent'),size=11,bold=True,ha='center')
        for key,xx,yy in [('W',490,193),('A',462,222),('S',490,222),('D',518,222)]:
            self.box(xx,yy,25,25,fill='#f1f5f7',radius=3)
            self.text(xx+12.5,yy+12.5,key,size=9,ha='center')
        self.text(507,282,t('平面速度 + 朝向','XZ velocity + heading'),size=9,ha='center',color=GRAY)
        self.box(632,149,230,166)
        self.text(747,170,t('静态 PFNN 高度场','Static PFNN heightfield'),size=11,bold=True,ha='center')
        self.terrain(644,195,204,99)
        self.text(747,301,t('完整分辨率 / 已知区域掩码','Full resolution / known mask'),size=8,ha='center',color=GRAY)

        self.box(907,162,224,139,fill='#f1f8f6',edge='#8fbbb0')
        self.text(1019,186,t('因果几何查询','Causal geometry queries'),size=11,bold=True,ha='center')
        self.text(1019,220,t('脚下与指令方向高度\n身体 / 指令 / 实际运动方向的三维采样',
            'Near-support + intent heights\nBody / intent / motion volumes'),size=8.5,ha='center')
        self.text(1019,275,t('距离 d · 法线 n · 有效性 v','Distance d / normal n / validity v'),size=9,ha='center',color=TEAL)
        self.line([(862,235),(907,235)])
        self.line([(507,315),(507,330),(887,330),(887,275),(907,275)],color=GRAY,lw=1)
        self.line([(382,201),(399,201),(399,140),(887,140),(887,197),(907,197)],color=GRAY,lw=1)

        self.box(1175,151,234,65,fill='#f0f6ee',edge='#a5bea4')
        self.text(1292,172,t('二维高度观察','2D height patches'),size=10,bold=True,ha='center')
        self.text(1292,198,'$O_h$:  2 x 9 x 9',size=10,ha='center',color=GREEN)
        self.box(1175,237,234,65,fill='#eaf6f4',edge='#8fbdb5')
        self.text(1292,257,t('三维距离观察','3D SDF observations'),size=10,bold=True,ha='center')
        self.text(1292,283,'$O_v$:  3 x 8 x 8 x 8',size=10,ha='center',color=TEAL)
        self.line([(1131,210),(1150,210),(1150,183),(1175,183)],color=GREEN)
        self.line([(1131,259),(1150,259),(1150,270),(1175,270)],color=TEAL)
        self.box(1453,237,289,65,fill='#eaf6f4',edge=TEAL,lw=1.3)
        self.text(1598,257,t('共享地形编码器','Shared terrain encoder'),size=11,bold=True,ha='center')
        self.text(1598,283,'$M_t$:  24 x 128',size=10,ha='center',color=TEAL)
        self.line([(1409,270),(1453,270)],color=TEAL)

        # Conditioning buses are color-coded and connect through explicit dots.
        self.line([(1598,302),(1598,370),(400,370)],color=TEAL,arrow=False)
        for x in (400,900,1520):
            self.dot(x,370,TEAL)
            self.line([(x,370),(x,478)],color=TEAL)
        self.label(1077,363,t('共享三维地形条件 M','Shared 3D terrain memory M'),color=TEAL,size=9)
        self.line([(1409,183),(1430,183),(1430,398),(305,398)],color=GREEN,arrow=False)
        for x in (305,790):
            self.dot(x,398,GREEN);self.line([(x,398),(x,478)],color=GREEN)
        self.label(1045,391,t('原生二维高度条件','Native 2D terrain conditioning'),color=GREEN,size=9)
        self.line([(90,315),(90,428),(1387,428)],color=GRAY,dash=True,arrow=False)
        for x in (200,690,1387):
            self.dot(x,428,GRAY);self.line([(x,428),(x,478)],color=GRAY,dash=True)
        self.label(102,416,t('动作历史 H','Motion context H'),color=GRAY,size=8.5)
        self.line([(507,315),(507,452),(455,452),(455,491)],color=GRAY,lw=1)
        self.label(522,445,t('当前速度指令','Current command'),size=8)

        self.text(46,355,t('(b) 地形条件动作生成','(b) Terrain-conditioned motion generation'),size=12,bold=True)
        self.box(150,478,310,186,fill='#eef4fb',edge=BLUE,lw=1.4)
        self.text(305,501,'Root',size=15,bold=True,ha='center',color=BLUE)
        self.text(305,529,t('预测整体位置、朝向与高度','Predict global motion and height'),size=9,ha='center')
        root=self.assets['world'][0,4:28,:3]
        xx=(root[:,2]-root[:,2].min())/max(float(np.ptp(root[:,2])),.001)
        yy=(root[:,1]-root[:,1].min())/max(float(np.ptp(root[:,1])),.001)
        self.ax.plot(203+xx*199,615-yy*40,color=BLUE,lw=2.1,zorder=7)
        self.ax.plot([199,408],[622,622],color='#b5c6dc',lw=.8,zorder=4)
        self.text(305,646,t('预测轨迹 R','Predicted root trajectory R'),size=9,ha='center',color=BLUE)

        self.box(590,478,355,186,fill='#f5f1f8',edge=PURPLE,lw=1.4)
        self.text(767,501,'Pose',size=15,bold=True,ha='center',color=PURPLE)
        self.text(767,528,t('4 轮掩码编码预测','4 masked-token prediction passes'),size=9,ha='center')
        self.token_grid(672,551,8,8,cell=8.7)
        self.text(645,594,t('码本头','Heads'),size=8,ha='right',color=GRAY)
        self.text(767,646,t('8 个时间位置 x 8 个码本头','8 temporal positions x 8 codebook heads'),size=9,ha='center',color=PURPLE)
        self.line([(460,558),(590,558)],color=BLUE,lw=1.6)
        self.label(525,543,'R',color=BLUE,size=11,ha='center')

        self.box(1000,508,170,127,fill='#faf6eb',edge='#cab383')
        self.text(1085,529,t('码本查表','Codebook lookup'),size=10,bold=True,ha='center',color=GOLD)
        self.token_grid(1042,546,10,8,cell=5.6,colors=['#d6bc80','#c79f59','#ead8ac','#b88a42'])
        self.text(1085,623,'8 heads / 10 codes / 32-D',size=7.5,ha='center',color=GOLD)
        self.line([(945,558),(1000,558)],color=PURPLE,lw=1.5)
        self.label(971,543,'Z',color=PURPLE,size=10,ha='center')

        self.box(1245,478,365,186,fill='#fcf5ec',edge=GOLD,lw=1.4)
        self.text(1427,501,t('条件动作解码器','Conditional motion decoder'),size=13,bold=True,ha='center',color=GOLD)
        self.text(1427,529,t('编码 + 历史 + 预测 Root + 地形','Codes + context + predicted Root + terrain'),size=8.8,ha='center')
        for i in range(4):
            self.box(1314+i*53,562-i*3,28+i*4,47+i*6,fill=['#e7d3af','#dbc191','#cda76b','#b88741'][i],edge='white',radius=2,lw=.5)
            if i<3:self.line([(1346+i*53,586),(1365+i*53,586)],color=GOLD,lw=.8)
        self.text(1427,645,t('每个时间位置解码为 4 帧','Decode 4 frames per temporal position'),size=9,ha='center',color=GOLD)
        self.line([(1170,558),(1245,558)],color=GOLD,lw=1.5)
        self.label(1207,543,t('连续嵌入','Embeddings'),size=8,ha='center',color=GOLD)
        self.line([(305,664),(305,696),(1427,696),(1427,664)],color=BLUE,lw=1.3)
        self.label(828,689,t('预测轨迹 R + 高度变化 / 垂直速度适配特征',
            'Predicted root trajectory R + height-change / vertical-speed adaptation'),color=BLUE,size=9,ha='center')

        # (c) Immutable output and actual playback history; arrows run right-to-left.
        self.text(46,737,t('(c) 完整动作播放与反馈','(c) Committed playback and feedback'),size=12,bold=True)
        self.box(1150,766,460,113,fill='#f4f7f9')
        self.text(1380,787,t('32 帧预测窗口','32-frame predicted window'),size=11,bold=True,ha='center')
        for i in range(32):
            fill='#cfd6df' if i<4 or i>=28 else '#65a39a'
            self.ax.add_patch(Rectangle((1174+i*12.8,811),10.8,20,facecolor=fill,edgecolor='white',lw=.3,zorder=6))
        self.text(1199,854,t('历史 4','Context 4'),size=8,ha='center',color=GRAY)
        self.text(1376,854,t('可播放未来 24','Playable future 24'),size=9,ha='center',color=TEAL)
        self.text(1559,854,t('尾部 4','Tail 4'),size=8,ha='center',color=GRAY)
        self.line([(1610,576),(1678,576),(1678,821),(1610,821)],color=GOLD,lw=1.5)

        self.box(758,766,302,113,fill='#f4f7f9')
        self.text(909,790,t('不可修改的八帧动作包','Immutable 8-frame packet'),size=10.5,bold=True,ha='center')
        for i in range(8):self.box(805+i*27,811,21,23,fill='#d6e8e4',edge='#91b8ae',radius=2,lw=.8)
        self.text(909,856,t('预取后核验接续状态','Prefetch + boundary verification'),size=8.5,ha='center',color=GRAY)
        self.line([(1150,823),(1060,823)],color=INK,lw=1.5)

        self.box(373,766,300,113,fill='#f4f7f9')
        self.text(523,787,t('官方 SOMA 蒙皮与播放','Official SOMA skinning / playback'),size=10,bold=True,ha='center')
        for i,index in enumerate((8,16,24)):self.skeleton(450+i*74,857,45,index,color=BLUE,alpha=.55+.20*i)
        self.text(523,868,t('18,056 顶点 / 30 fps','18,056 vertices / 30 fps'),size=8,ha='center',color=GRAY)
        self.line([(758,823),(673,823)],color=INK,lw=1.5)

        self.box(52,766,240,113,fill='#edf4fa',edge='#9bb1c9')
        self.text(172,794,t('播放确认与历史重编码','Playback ACK + history re-encoding'),size=9.4,bold=True,ha='center')
        self.text(172,826,t('仅更新最新四帧已播放状态','Update the last 4 played frames'),size=8.8,ha='center')
        self.text(172,855,t('FK 一致性检查','FK-consistent feedback'),size=8.5,ha='center',color=BLUE)
        self.line([(373,823),(292,823)],color=INK,lw=1.5)
        self.line([(52,823),(17,823),(17,230),(52,230)],color=BLUE,lw=1.5)

        self.text(45,906,t('已实现调参：重规划 1 / 2 / 4 / 8 / 16 / 24 帧；小于 8 帧时使用动作包内已生成的前缀反馈。',
            'Implemented tuning: replan every 1 / 2 / 4 / 8 / 16 / 24 frames; sub-packet feedback uses immutable generated prefixes.'),
            size=9,color=GRAY)

        # (d) Three-stage completed training, with the frozen codebook dependency explicit.
        self.box(30,933,1740,207,fill='#fcfcfd',edge=LINE,radius=9,dash=(0,(5,3)))
        self.text(46,954,t('(d) 离线分阶段训练','(d) Offline staged training'),size=12,bold=True)
        cards=[(52,260,'#f3f6f8',LINE,t('PFNN + PARC + BONES','PFNN + PARC + BONES'),
            t('SOMA30 重定向 / 30 Hz\n动作与静态地形配对', 'SOMA30 retargeting / 30 Hz\nMotion + static terrain pairs')),
            (363,310,'#eef4fb',BLUE,t('阶段 1：Root','Stage 1: Root'),
            t('整体运动监督\n共享地形编码器 + Root 适配器','Root-motion supervision\nShared terrain encoder + Root adapters')),
            (730,310,'#fcf5ec',GOLD,t('阶段 2：Codec','Stage 2: Codec'),
            t('编码器 → 多头量化 → 解码器\n动作重建 + 原生辅助目标','Encoder > multi-head VQ > decoder\nMotion reconstruction + native objectives')),
            (1100,310,'#f5f1f8',PURPLE,t('阶段 3：Pose','Stage 3: Pose'),
            t('冻结最终 Codec 与码本\n生成新标签，训练掩码编码预测','Freeze the final Codec / codebooks\nCreate labels; train masked-token prediction')),
            (1468,274,'#edf5f0',GREEN,t('部署同一完整模型','Deploy the complete model'),
            t('Root + Pose + Codec\n全部地形适配器','Root + Pose + Codec\nAll terrain adapters'))]
        for x,w,fill,color,title,body in cards:
            self.box(x,978,w,111,fill=fill,edge=color,lw=1)
            self.text(x+w/2,1001,title,size=10.5,bold=True,ha='center',color=color if color!=LINE else INK)
            self.text(x+w/2,1045,body,size=8.7,ha='center')
        for start,end in ((312,363),(673,730),(1040,1100),(1410,1468)):
            self.line([(start,1034),(end,1034)],color=GRAY,lw=1,dash=True)
        self.text(52,1114,t('训练可使用源 Root 教师；在线仅使用预测 Root。未来源姿态不进入在线输入。',
            'Source Root teachers are training-only; online generation uses predicted Root and no source future poses.'),
            size=9,color=GRAY)
        self.text(34,1160,t('实线：在线数据流    虚线：状态条件 / 离线训练依赖    不含运行时 IK、锁脚、碰撞投影或输出平滑',
            'Solid: online flow     Dashed: context / training dependencies     No runtime IK, foot locking, collision projection or output smoothing'),
            size=8.5,color=GRAY)

    def save(self,output):
        stem=output/f'terrain_aware_motionbricks_method_{self.language}'
        for extension in ('png','svg','pdf'):
            self.figure.savefig(stem.with_suffix('.'+extension),dpi=240,facecolor='white',
                metadata={'Creator':'MotionBricks implementation diagram'})
        plt.close(self.figure)
        return [str(stem.with_suffix('.'+ext).relative_to(ROOT)) for ext in ('png','svg','pdf')]


@torch.inference_mode()
def main(args):
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    for language in ('zh','en'):
        if (args.output/f'terrain_aware_motionbricks_method_{language}.png').exists():
            raise FileExistsError('Preserve the previous figure; choose a new output directory')
    scene=ROOT/'out/pfnn_mixed_20261015/pfnn_map_catalog_v1/003/scene.npz'
    recording=ROOT/'out/pfnn_mixed_20261015/terrain_refresh_diagnosis_v1/forward_walk-24.pt'
    weights=ROOT/'out/pfnn_mixed_20261015/pfnn_map_catalog_v1/003/weights.pt'
    saved=torch.load(weights,map_location='cpu',mmap=True,weights_only=False)
    shape=list(saved['model_state']['codec.quantizer.vq._codebook.embed'].shape)
    if shape!=[8,10,32]:raise ValueError('Diagram labels require the current eight-head codebook')
    world=torch.load(recording,map_location='cpu',weights_only=False)['world']
    rep=build_native_representation()
    pose=rep.inverse(world,False,return_all=False)
    with np.load(scene,allow_pickle=False) as data:
        assets={key:data[key].copy() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
    assets=dict(height=assets['heightfield_y_m'],origin=assets['min_point_world_xyz_m'],
        dx=float(assets['dx_m']),world=world.numpy(),joints=pose['posed_joints'][0].numpy(),
        parents=rep.skeleton.joint_parents.tolist())
    outputs=[]
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False})
    for language in ('zh','en'):
        diagram=Diagram(language,assets);diagram.build();outputs.extend(diagram.save(args.output))
    (args.output/'figure_receipt.json').write_text(json.dumps(dict(
        schema='motionbricks.method_figure.v1',outputs=outputs,implementation_only=True,
        codebook_heads=shape[0],codes_per_head=shape[1],embedding_dimensions=shape[2],
        temporal_positions=8,frames_per_position=4,pose_passes=4,history_frames=4,
        packet_frames=8,default_played_frames=24,fps=30,
        source_scene_sha256=sha(scene),source_recording_sha256=sha(recording),
        figure_source_sha256=sha(Path(__file__)),actual_optimizer_updates=0,
        illustrations='Terrain tiles and skeletons from existing map003 diagnostic assets. Token colors are schematic.',
        no_quality_approval_implied=True,proposed_phase_student_physics_modules_included=False),indent=2))
    print(json.dumps(dict(outputs=outputs,codebook_shape=shape)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/pfnn_method_20261017')
    main(parser.parse_args())
