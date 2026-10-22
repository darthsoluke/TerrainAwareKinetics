"""Independent real-clip cursors and exact penetration for joint-SDF learning."""
from collections import Counter

import numpy as np
import torch

from .terrain_control import SoftCommandState,demonstration_controls
from .terrain_model_v3 import pack_scene_chunks
from .terrain_scene_frame import HorizontalSceneFrame,known_history_centers
from .terrain_batched_geometry import PackedHeightfields
from .joint_sdf_observation import HeightfieldPrismSDF
from .terrain_response_loss import heightfield_gap


def episode_segments(frame_count,maximum=13):
    if frame_count<33:raise ValueError('A real 33-frame interval is required')
    return min(maximum,1+(frame_count-33)//8)


@torch.no_grad()
def verified_lateral_offsets(points,scene,offsets,*,tolerance=1e-5):
    """Keep only source translations preserving all sampled terrain heights.

    This certifies a particular source's raster correspondence, not arbitrary
    scene symmetry or permission to adjust generated motion at runtime.
    """
    reference,known=heightfield_gap(points,scene)
    if not known.all():raise ValueError('Reference geometry must lie within measured terrain')
    accepted=[];report=[]
    for offset in offsets:
        if not np.isfinite(offset):raise ValueError('Finite source translation required')
        moved=points.clone();moved[...,2]+=float(offset)
        gap,valid=heightfield_gap(moved,scene)
        error=float((gap-reference)[known & valid].abs().max()) if (known & valid).any() else None
        passed=bool(valid.all() and error is not None and error<=tolerance)
        if passed:accepted.append(float(offset))
        report.append({'offset_world_z_m':float(offset),'all_known':bool(valid.all()),
            'max_surface_height_change_m':error,'accepted':passed})
    return accepted,report


@torch.no_grad()
def approach_intent(item):
    """Infer weak cruise intent only from a real straight, flat approach.

    Actual speed on a stair is an outcome, not necessarily requested speed.
    Ambiguous, turning or mid-terrain episodes retain ordinary source labels.
    """
    if not item['row']['origin'].startswith('parc'):return None
    world=item['world'];gap,known=heightfield_gap(world[:,:3],item['scene']);ground=world[:,1]-gap
    if not known.all() or ground[:16].max()-ground[:16].min()>.03:return None
    if (ground-ground[3]).abs().max()<.15:return None
    edge=int(torch.nonzero((ground-ground[3]).abs()>.03)[0,0])
    approach_end=max(15,edge-8)
    delta=world[approach_end,[0,2]]-world[3,[0,2]]
    if delta.norm()<.2:return None
    direction=delta/delta.norm();side=torch.stack((direction[1],-direction[0]))
    if ((world[:,[0,2]]-world[3,[0,2]])*side).sum(-1).abs().max()>.35:return None
    facing=world[3:approach_end,3:5].mean(0);facing/=facing.norm().clamp_min(1e-6)
    return torch.cat((delta/((approach_end-3)/30),facing))


class IndependentEpisodes:
    def __init__(self,data,weights,batch,rep,device,seed):
        self.data=data;self.weights=np.asarray(weights,dtype=np.float64)
        self.weights/=self.weights.sum();self.batch=batch;self.rep=rep;self.device=device
        self.rng=np.random.default_rng(seed);self.draw=0;self.slots=[None]*batch
        self.command=SoftCommandState();self.counts=Counter();self.packed=None

    def _restore_item(self,slot):
        slot['item']=self.data[(slot['index'],slot['draw'])]

    @torch.no_grad()
    def window(self):
        changed=False;new=[]
        for row,slot in enumerate(self.slots):
            if slot is None or slot['position']==slot['segments']:
                if slot is not None:self.counts['completed_depth:'+str(slot['segments'])]+=1
                index=int(self.rng.choice(len(self.data),p=self.weights));draw=self.draw;self.draw+=1
                item=self.data[(index,draw)]
                segments=episode_segments(len(item['world']))
                length=33+(segments-1)*8
                start=int(self.rng.integers(len(item['world'])-length+1))
                slot={'index':index,'draw':draw,'start':start,'position':0,'segments':segments,
                    'yaw':float(self.rng.uniform(-np.pi,np.pi)),'item':item,'history':None,
                    'approach_control':approach_intent(dict(item,world=item['world'][start:start+length]))}
                self.slots[row]=slot;changed=True;new.append(row)
                origin=item['row']['origin'];self.counts['episode:'+origin]+=1
        scenes=[slot['item']['scene'] for slot in self.slots]
        if changed or self.packed is None:self.packed=pack_scene_chunks(scenes,self.device)
        windows=torch.stack([s['item']['world'][s['start']+s['position']*8:s['start']+s['position']*8+33]
            for s in self.slots]).to(self.device)
        physical_history=torch.stack([windows[i,:4] if s['history'] is None else s['history'].to(self.device)
            for i,s in enumerate(self.slots)])
        known=known_history_centers(physical_history,self.packed)
        resets=~known
        self.counts['unknown_history_resets']+=int(resets.sum())
        # A missing observation can restart training only at the factual time.
        # Evaluation never uses this repair and retains the incomplete rollout.
        physical_history=torch.where(known[:,None,None],physical_history,windows[:,:4])
        if not known_history_centers(physical_history,self.packed).all():
            raise ValueError('Factual training source is outside observed terrain')
        yaw=windows.new_tensor([s['yaw'] for s in self.slots])
        frame=HorizontalSceneFrame(yaw,windows[:,0,[0,2]])
        source=frame.motion(self.rep,windows);history=frame.motion(self.rep,physical_history)
        controls=[]
        for i,slot in enumerate(self.slots):
            item=slot['item'];category=item['row'].get('control_category')
            reference=frame.row(i).motion(self.rep,item['world'][None].to(self.device)) if category=='start' else None
            controls.append(demonstration_controls(source[i:i+1],32,category=category,reference=reference))
            cruise=slot.get('approach_control')
            if cruise is not None:
                cruise=cruise.to(self.device)[None]
                velocity=frame.row(i).vectors(cruise[:,:2])
                yaw=torch.atan2(cruise[:,3],cruise[:,2])+frame.yaw[i]
                controls[-1]=torch.cat((velocity,yaw.cos()[:,None],yaw.sin()[:,None]),-1)
                self.counts['weak_approach_intent_primitives']+=1
        if self.command.stop_goal is not None:
            # Store persistent stop requests in the unchanged physical frame.
            self.command.stop_goal=frame.points(self.command.stop_goal.to(self.device))
            self.command.was_moving=self.command.was_moving.to(self.device)
            restart=resets.clone();restart[new]=True
            self.command.stop_goal[restart]=history[restart,3][:,[0,2]]
            self.command.was_moving[restart]=True
        intent=self.command.from_controls(history,torch.cat(controls),8)
        self.command.stop_goal=frame.points(self.command.stop_goal,inverse=True)
        for slot in self.slots:
            row=slot['item']['row'];category=row.get('control_category') or row.get('terrain_family','unknown')
            self.counts['primitive:'+row['origin']+':'+str(category)]+=1
            self.counts['feedback_position:'+str(slot['position'])]+=1
        return source,windows,history,physical_history,intent,scenes,frame

    def advance(self,physical_history):
        for row,slot in enumerate(self.slots):
            slot['history']=physical_history[row].detach();slot['position']+=1

    def state_dict(self):
        return {'slots':[{k:(v.detach().cpu() if torch.is_tensor(v) else v)
                    for k,v in slot.items() if k!='item'} for slot in self.slots],
            'rng':self.rng.bit_generator.state,'draw':self.draw,'counts':dict(self.counts),
            'stop_goal':None if self.command.stop_goal is None else self.command.stop_goal.detach().cpu(),
            'was_moving':None if self.command.was_moving is None else self.command.was_moving.cpu()}

    def load_state_dict(self,state):
        if len(state['slots'])!=self.batch:raise ValueError('Exact replay requires identical batch/world size')
        self.slots=state['slots'];self.rng.bit_generator.state=state['rng'];self.draw=state['draw']
        self.counts=Counter(state['counts']);self.command.stop_goal=state['stop_goal'];self.command.was_moving=state['was_moving']
        for slot in self.slots:self._restore_item(slot)
        self.packed=None


class ExactPenetrationBatch:
    """Exact negative SDF; positive values are height gaps, not exterior SDF."""
    def __init__(self,scenes,device,fields,packed=None):
        self.heights=PackedHeightfields(scenes,device,packed=packed);self.fields=fields

    def query(self,points):
        gap,known=self.heights.query(points);rows=[]
        for index,field in enumerate(self.fields):
            inside=known[index] & (gap[index]<0)
            result=gap[index].clamp_min(0)
            if inside.any():
                distance,_,valid=field.query(points[index][inside])
                result=result.masked_scatter(inside,torch.where(valid,distance,gap[index][inside]))
            rows.append(result)
        return torch.stack(rows),known
