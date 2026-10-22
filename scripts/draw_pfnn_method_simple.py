"""A five-step, illustrated method figure for the current controller."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.draw_pfnn_method_figure import Diagram,ROOT,INK,GRAY,LINE,TEAL,BLUE,PURPLE,GOLD,GREEN,sha


class SimpleMethod(Diagram):
    def __init__(self,language,assets):
        super().__init__(language,assets)
        self.figure.set_size_inches(18,10)
        self.ax.set_ylim(1000,0)

    def build(self):
        t=self.tr
        self.text(40,48,t('从按键和地形，到连续的角色动作',
            'From User Input and Terrain to Continuous Motion'),size=22,bold=True)
        self.text(40,87,t('Terrain-aware MotionBricks · 当前方法',
            'Terrain-aware MotionBricks / Current implementation'),size=11,color=GRAY)

        xs=[40,390,740,1090,1440]
        colors=[BLUE,TEAL,BLUE,PURPLE,GOLD]
        titles=[t('输入与当前状态','Input + current state'),t('观察附近地形','Observe nearby terrain'),
            t('预测身体怎么走','Plan body movement'),t('生成全身动作','Generate full-body motion'),
            t('让角色动起来','Animate the character')]
        subtitles=[t('方向、速度和刚才的动作','Direction, speed, recent motion'),
            t('脚下、前方和实际移动方向','Underfoot, ahead, actual travel'),
            t('Root：位置、朝向、高度','Root: position, heading, height'),
            t('Pose + 动作解码器','Pose + motion decoder'),t('SOMA 蒙皮 / 30 fps','SOMA skinning / 30 fps')]
        fills=['#f1f6fc','#edf7f4','#f1f6fc','#f6f2f9','#fcf6ed']
        for i,(x,color,title,subtitle,fill) in enumerate(zip(xs,colors,titles,subtitles,fills)):
            self.box(x,174,300,340,fill=fill,edge=color,radius=9,lw=1.2)
            self.ax.add_patch(Circle((x+27,201),13,facecolor=color,edgecolor='none',zorder=4))
            self.text(x+27,201,str(i+1),size=10.5,bold=True,color='white',ha='center')
            self.text(x+150,238,title,size=14,bold=True,ha='center',color=color)
            self.text(x+150,274,subtitle,size=9.5,ha='center',color=GRAY)
            if i<4:self.line([(x+300,353),(xs[i+1],353)],color=GRAY,lw=1.8)

        # Input: keyboard command and an actual four-frame pose prefix.
        for key,xx,yy in [('W',151,304),('A',119,337),('S',151,337),('D',183,337)]:
            self.box(xx,yy,28,28,fill='white',edge='#b9c7d8',radius=3)
            self.text(xx+14,yy+14,key,size=10,ha='center',color=BLUE)
        for i in range(4):self.skeleton(91+i*63,457,60,i,color=BLUE,alpha=.45+.18*i)
        self.text(190,486,t('最近 4 帧动作','The last 4 motion frames'),size=10,ha='center')

        # Terrain: a cropped real packaged map, with schematic query markers.
        self.terrain(400,319,275,149)
        for dx,dz in ((0,0),(8,6),(16,12),(24,18),(32,24)):
            self.ax.scatter([498+dx,524+dx],[414-dz,424-dz],s=13,color=TEAL,zorder=9)
        self.text(540,486,t('地面有多高，哪里是斜坡或凸起','Heights, slopes, and raised ground'),size=9.3,ha='center')

        # A root path illustrates the predicted movement plan, not a measured result.
        ground_x=np.linspace(768,1012,100)
        curve=(np.clip((ground_x-806)/154,0,1))
        curve=curve*curve*(3-2*curve)
        ground_y=443-52*curve
        self.ax.fill_between(ground_x,ground_y,460,color='#d6e1dc',zorder=3)
        self.ax.plot(ground_x,ground_y,color='#8ca498',lw=1.2,zorder=4)
        self.ax.plot(ground_x[8:-5],ground_y[8:-5]-53,color=BLUE,lw=2.4,zorder=7)
        chosen=[14,40,65,89]
        self.ax.scatter(ground_x[chosen],ground_y[chosen]-53,s=26,color=BLUE,zorder=8)
        self.line([(ground_x[87],ground_y[87]-53),(ground_x[94],ground_y[94]-53)],color=BLUE,lw=1.8)
        self.text(890,486,t('先预测身体整体的移动','Predict the body movement first'),size=10,ha='center')

        # Full-body generation uses existing decoded poses for the illustration.
        for i,index in enumerate((12,32,56)):
            self.skeleton(1140+i*94,452,126,index,color=PURPLE,alpha=.60+.18*i)
        self.text(1240,486,t('把移动计划变成连续姿态','Turn the plan into continuous poses'),size=9.5,ha='center')

        # Browser-like playback window and character, deliberately without extra UI.
        self.box(1457,307,266,151,fill='white',edge='#d7c6a9',radius=5)
        self.ax.plot([1458,1722],[326,326],color='#e3d8c8',lw=.8,zorder=5)
        for i in range(3):self.ax.add_patch(Circle((1469+i*10,316),2.3,facecolor='#c6b596',edgecolor='none',zorder=5))
        self.ax.add_patch(Rectangle((1458,407),264,50,facecolor='#edf1ed',edgecolor='none',zorder=3))
        self.skeleton(1585,439,99,40,color=GOLD)
        self.text(1590,486,t('按时间播放，得到新的动作状态','Play in time; obtain the new state'),size=9.5,ha='center')

        self.line([(540,514),(540,563),(1240,563),(1240,514)],color=TEAL,lw=1.4)
        self.label(890,588,t('地形信息同时参与整体移动和全身姿态生成',
            'Terrain informs both body movement and full-body pose'),color=TEAL,size=11,ha='center')

        # One easily followed feedback loop instead of a network of crossing buses.
        self.line([(1590,514),(1590,658),(190,658),(190,514)],color=BLUE,lw=1.8)
        self.label(890,658,t('播放后的最近 4 帧，作为下一轮预测的起点',
            'The last 4 played frames become the starting point for the next prediction'),
            color=BLUE,size=12,ha='center')
        self.text(890,710,t('每次只生成接下来一小段，然后根据新状态继续。',
            'Generate a short segment, then continue from the updated state.'),size=12,ha='center',color=GRAY)

        self.box(40,768,1700,172,fill='#fafbfc',edge=LINE,radius=8)
        self.text(58,794,t('模型如何学会这些动作？','How is the model trained?'),size=12,bold=True)
        self.text(58,837,'PFNN + PARC + BONES',size=12,bold=True)
        self.text(58,872,t('统一骨架，并与地形配对','Retarget and pair motions with terrain'),size=9,color=GRAY)
        stages=[(426,BLUE,t('学身体怎么移动','Learn body movement'),'Root'),
            (849,GOLD,t('学动作的压缩与还原','Learn motion compression / decoding'),'Codec'),
            (1273,PURPLE,t('学根据当前条件生成动作','Learn motion generation from conditions'),'Pose')]
        for i,(x,color,label,module) in enumerate(stages):
            self.line([(360 if i==0 else stages[i-1][0]+365,854),(x,854)],color=GRAY,lw=1.15)
            self.box(x,817,365,75,fill='white',edge=color,radius=5)
            self.text(x+182,842,label,size=10.5,bold=True,color=color,ha='center')
            self.text(x+182,870,module,size=10,ha='center',color=GRAY)
        self.text(58,919,t('训练用真实动作作答案；操控时，后续动作由模型自己预测。',
            'Training uses recorded motion as supervision; runtime predicts the motion itself.'),size=10,color=GRAY)
        self.text(40,973,t('当前实现示意 · 未加入后续计划中的相位模型、轻量学生或物理控制层',
            'Current implementation; proposed phase models, lightweight students and physics controllers are not included.'),
            size=8.5,color=GRAY)


@torch.inference_mode()
def main(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    scene=ROOT/'out/pfnn_mixed_20261015/pfnn_map_catalog_v1/003/scene.npz'
    recording=ROOT/'out/pfnn_mixed_20261015/terrain_refresh_diagnosis_v1/forward_walk-24.pt'
    world=torch.load(recording,map_location='cpu',weights_only=False)['world']
    rep=build_native_representation()
    pose=rep.inverse(world,False,return_all=False)
    with np.load(scene,allow_pickle=False) as data:
        assets=dict(height=data['heightfield_y_m'].copy(),origin=data['min_point_world_xyz_m'].copy(),
            dx=float(data['dx_m']))
    assets.update(world=world.numpy(),joints=pose['posed_joints'][0].numpy(),parents=rep.skeleton.joint_parents.tolist())
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False})
    outputs=[]
    for language in ('zh','en'):
        diagram=SimpleMethod(language,assets);diagram.build();outputs.extend(diagram.save(output))
    (output/'figure_receipt.json').write_text(json.dumps(dict(outputs=outputs,implementation_only=True,
        actual_optimizer_updates=0,source_recording_sha256=sha(recording),source_scene_sha256=sha(scene),
        source_sha256=sha(Path(__file__)),drawing_helpers_sha256=sha(ROOT/'scripts/draw_pfnn_method_figure.py'),
        illustrations='Terrain and pose miniatures reuse existing assets; the path is schematic.',
        quality_approval_implied=False),indent=2))
    print(json.dumps(dict(outputs=outputs)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/pfnn_method_simple_20261017')
    main(parser.parse_args())
