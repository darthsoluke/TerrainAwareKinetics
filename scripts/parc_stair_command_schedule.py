"""Issue causal move/zero-direction/resume requests, never edit generated poses."""

import math

import torch


class StairCommandSchedule:
    def __init__(self,nodes,course,batch_size=8,fps=30,stride=7):
        self.nodes=nodes;self.course=course;self.fps=fps;self.stride=stride
        self.axis=(nodes[-1,:2]-nodes[0,:2]);self.axis/=self.axis.norm()
        self.side=torch.stack((-self.axis[1],self.axis[0]))
        self.phases=[next(s for s in course['route_segments'] if s['stage']==name) for name in ('ascent','descent')]
        self.phase=[0]*batch_size;self.remaining=[0]*batch_size;self.triggered=[False]*batch_size
        self.calls=[]

    def transform(self,kw):
        kw=dict(kw); previous=kw['prev_frames'];batch=len(previous.root_pos)
        if batch!=len(self.phase):raise ValueError('Candidate identities changed')
        if not self.calls:
            previous=previous.get_copy(previous.root_pos.device)
            along=previous.root_pos.new_tensor([-.3,-.2,-.1,0.,.1,.2,.3,.4])
            lateral=previous.root_pos.new_tensor([-.1,0.,.1,0.,-.1,0.,.1,0.])
            previous.root_pos[:,-1,:2]+=along[:,None]*self.axis+lateral[:,None]*self.side
            kw['prev_frames']=previous
            target=previous.root_pos[:,-1].clone();target[:,:2]+=self.axis
        else:
            target=kw['target_world_pos'].expand(batch,-1).clone()
        labels=[]
        for row in range(batch):
            point=previous.root_pos[row,-1,:2]
            if self.phase[row]<2 and not self.triggered[row]:
                phase=self.phases[self.phase[row]]
                first=self.nodes[phase['path_node_start'],:2];last=self.nodes[phase['path_node_end'],:2]
                fraction=float((point-first) @ self.axis / ((last-first) @ self.axis))
                trigger=.25+.05*(row%3)
                if fraction>=trigger:
                    seconds=(1.6,2.,2.4,2.8)[row%4]
                    self.remaining[row]=math.ceil(seconds*self.fps/self.stride)
                    self.triggered[row]=True
            holding=self.phase[row]<2 and self.remaining[row]>0
            label=('ascent_stop','descent_stop')[self.phase[row]] if holding else 'move'
            if holding:
                target[row,:2]=point
                self.remaining[row]-=1
                if self.remaining[row]==0:
                    self.phase[row]+=1;self.triggered[row]=False
            labels.append(label)
        kw['target_world_pos']=target
        self.calls.append(dict(call=len(self.calls),labels=labels,
            observed_root_xy=previous.root_pos[:,-1,:2].detach().cpu().tolist(),
            requested_target_xy=target[:,:2].detach().cpu().tolist()))
        return kw


def stop_response_report(frames,commands,terrain,nodes,course,fps):
    """Kinematic pre-screen only; actual SOMA contacts still need verification."""
    import numpy as np
    stopped=np.linalg.norm(commands,axis=-1)<1e-6
    runs=[];start=None
    for i,value in enumerate(np.r_[stopped,False]):
        if value and start is None:start=i
        if not value and start is not None:
            if i-start>=30:runs.append((start,i))
            start=None
    root=frames.root_pos.detach().cpu().numpy()
    axis=(nodes[-1,:2]-nodes[0,:2]).detach().cpu().numpy();axis/=np.linalg.norm(axis)
    phases=[next(s for s in course['route_segments'] if s['stage']==stage) for stage in ('ascent','descent')]
    reports=[]
    for index,(start,end) in enumerate(runs):
        tail=root[max(start,end-15):end,:2]
        speed=np.linalg.norm(np.diff(tail,axis=0),axis=-1)*fps
        drift=float(np.linalg.norm(tail[-1]-tail[0]))
        after=min(len(root)-1,end+29)
        resume=float((root[after,:2]-root[end-1,:2]) @ axis)
        phase=phases[min(index,1)]
        a=nodes[phase['path_node_start'],:2].detach().cpu().numpy()
        b=nodes[phase['path_node_end'],:2].detach().cpu().numpy()
        progress=float((tail[-1]-a) @ axis);length=float((b-a) @ axis)
        inside=.15<progress<length-.15
        settled=float(np.quantile(speed,.9))<=.15 and drift<=.08
        reports.append(dict(phase=phase['stage'],start_frame=start,end_frame=end-1,
            final_half_second_speed_p90_mps=float(np.quantile(speed,.9)),final_half_second_drift_m=drift,
            stopped_inside_flight=inside,resume_progress_m=resume,kinematic_stop_response=settled and inside and resume>=.25))
    return dict(stops=reports,kinematic_stops_pass=len(reports)==2 and all(r['kinematic_stop_response'] for r in reports),
        scope='No motion modified. Zero direction is a requested stop, not proof of a stopped body; no speed label claimed. SOMA contacts and penetration must still pass.')
