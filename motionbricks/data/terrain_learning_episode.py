"""Replayable factual sequence and generated history across optimizer updates."""

import torch

from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_model_v3 import pack_scene_chunks
from motionbricks.data.terrain_scene_frame import HorizontalSceneFrame,known_history_centers


class TerrainLearningEpisode:
    def __init__(self,items,starts,frames,segments,representation,device,*,yaw=None,far=12.,width=2.):
        self.items=items;self.starts=list(map(int,starts));self.frames=frames;self.segments=segments
        self.far=far;self.width=width;self.position=0;self.rep=representation;self.device=device
        required=frames+1+(segments-1)*8
        windows=[item['world'][start:start+required] for item,start in zip(items,self.starts)]
        if any(len(window)!=required for window in windows):raise ValueError('Episode needs complete source intervals; no padding')
        self.physical_source=torch.stack(windows).to(device)
        self.frame=None if yaw is None else HorizontalSceneFrame(torch.as_tensor(yaw,device=device,dtype=self.physical_source.dtype),
            self.physical_source[:,0,[0,2]].detach())
        self.source=self.physical_source if self.frame is None else self.frame.motion(self.rep,self.physical_source)
        self.model_items=[dict(item,world=self.frame.row(i).motion(self.rep,item['world'][None].to(device))[0])
            if self.frame is not None and item['row'].get('control_category')=='start' else item for i,item in enumerate(items)]
        self.scenes=[i['scene'] for i in items];self.identities=[i['row']['source_id'] for i in items]
        self.packed_scene=pack_scene_chunks(self.scenes,device)
        self.history=self.source[:,:4].detach();self.command_state=SoftCommandState()

    @torch.no_grad()
    def window(self,teacher):
        if not 0<=self.position<self.segments:raise ValueError('Episode is exhausted')
        offset=self.position*8
        window=self.source[:,offset:offset+self.frames+1]
        physical_window=self.physical_source[:,offset:offset+self.frames+1]
        if teacher:self.history=window[:,:4].detach()
        physical_history=self.history if self.frame is None else self.frame.motion(self.rep,self.history,inverse=True)
        known=known_history_centers(physical_history,self.packed_scene)
        resets=int((~known).sum())
        if resets:
            if teacher:raise ValueError('Factual source history lacks an observed scene center')
            self.history=torch.where(known[:,None,None],self.history,window[:,:4].detach())
            physical_history=torch.where(known[:,None,None],physical_history,physical_window[:,:4])
            if not known_history_centers(physical_history,self.packed_scene).all():raise ValueError('Factual restart lacks observed terrain')
        return window,physical_window,physical_history,resets

    def advance(self,history):
        if history.shape!=self.history.shape:raise ValueError('Keep exactly four factual generated FK frames')
        self.history=history.detach();self.position+=1

    def state_dict(self):
        return {'requests':[(i['sampling_index'],i['sampling_draw']) for i in self.items],
            'starts':self.starts,'frames':self.frames,'segments':self.segments,'position':self.position,
            'yaw':None if self.frame is None else self.frame.yaw.detach().cpu(),
            'far':self.far,'width':self.width,'history':self.history.detach().cpu(),
            'stop_goal':None if self.command_state.stop_goal is None else self.command_state.stop_goal.detach().cpu(),
            'was_moving':None if self.command_state.was_moving is None else self.command_state.was_moving.detach().cpu()}

    @classmethod
    def restore(cls,state,data,representation,device):
        items=[data[tuple(request)] for request in state['requests']]
        episode=cls(items,state['starts'],state['frames'],state['segments'],representation,device,
            yaw=state['yaw'],far=state['far'],width=state['width'])
        episode.position=state['position'];episode.history=state['history'].to(device)
        if state['stop_goal'] is not None:episode.command_state.stop_goal=state['stop_goal'].to(device)
        if state['was_moving'] is not None:episode.command_state.was_moving=state['was_moving'].to(device)
        return episode
