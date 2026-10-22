"""Exact signed queries for known heightfield solids and 15 SOMA landmarks."""

import numpy as np
import torch


JOINT_SDF_NAMES=('LeftFoot','LeftToeBase','LeftShin','RightFoot','RightToeBase','RightShin',
    'LeftHand','LeftHandMiddleEnd','LeftForeArm','RightHand','RightHandMiddleEnd','RightForeArm',
    'Head','Chest','Hips')


def _rectangles(heights,known):
    active={};result=[]
    for z,row in enumerate(heights):
        edges=np.r_[0,np.flatnonzero((row[1:]!=row[:-1])|(known[z,1:]!=known[z,:-1]))+1,len(row)]
        current={}
        for x0,x1 in zip(edges[:-1],edges[1:]):
            if not known[z,x0]:continue
            key=(int(x0),int(x1),float(row[x0]))
            current[key]=active.pop(key,(z,z+1))
            current[key]=(current[key][0],z+1)
        result.extend((x0,x1,z0,z1,h) for (x0,x1,h),(z0,z1) in active.items())
        active=current
    result.extend((x0,x1,z0,z1,h) for (x0,x1,h),(z0,z1) in active.items())
    return result


class HeightfieldPrismSDF:
    """Union of downward solid prisms / upward air prisms, without voxel bias.

    Equal-height cells merge into rectangles without changing their geometry.
    Missing cells and map exterior are unknown, not free-space observations.
    """
    def __init__(self,scene):
        hf=torch.as_tensor(scene['heightfield_y_m']).detach().cpu().numpy()
        self.origin=np.asarray(torch.as_tensor(scene['min_point_world_xyz_m']).cpu(),dtype=np.float32)
        self.dx=float(scene['dx_m']);self.known=np.isfinite(hf)
        if 'heightfield_valid' in scene:self.known&=np.asarray(torch.as_tensor(scene['heightfield_valid']).cpu(),dtype=bool)
        if hf.ndim!=2 or not self.known.any() or self.dx<=0:raise ValueError('Known positive-spacing heightfield required')
        self.heights=np.nan_to_num(hf).astype(np.float32)
        self.rectangles=_rectangles(self.heights,self.known)
        self.holes=_rectangles(np.zeros_like(self.heights),~self.known)
        self.cache={}

    def _tensors(self,points):
        key=(points.device,points.dtype)
        if key not in self.cache:
            def boxes(rows):
                a=points.new_tensor(rows).reshape(-1,5)
                center=torch.stack((self.origin[0]+(a[:,0]+a[:,1]-1)*self.dx/2,
                    self.origin[2]+(a[:,2]+a[:,3]-1)*self.dx/2),-1)
                half=torch.stack(((a[:,1]-a[:,0])*self.dx/2,(a[:,3]-a[:,2])*self.dx/2),-1)
                return center,half,a[:,4]
            self.cache[key]=(boxes(self.rectangles),boxes(self.holes),points.new_tensor(self.heights),
                torch.as_tensor(self.known,device=points.device),points.new_tensor(self.origin))
        return self.cache[key]

    def query(self,points):
        if points.shape[-1]!=3 or not torch.isfinite(points).all():raise ValueError('Finite XYZ queries required')
        boxes,holes,hf,known,origin=self._tensors(points)
        center,half,height=boxes;p=points.reshape(-1,3)
        cell=torch.floor((p[:,[0,2]].detach()-origin[[0,2]])/self.dx+.5).long();x,z=cell.unbind(-1)
        valid=(x>=0)&(z>=0)&(x<hf.shape[1])&(z<hf.shape[0])
        valid&=known[z.clamp(0,hf.shape[0]-1),x.clamp(0,hf.shape[1]-1)]
        inside=p[:,1].detach()<hf[z.clamp(0,hf.shape[0]-1),x.clamp(0,hf.shape[1]-1)]
        distances=[];normals=[];certainty=[]
        for start in range(0,len(p),512):
            q=p[start:start+512];interior=inside[start:start+512]
            with torch.no_grad():
                best=torch.full((len(q),),torch.inf,device=q.device,dtype=q.dtype)
                index=torch.zeros(len(q),device=q.device,dtype=torch.long)
                for j in range(0,len(center),1024):
                    horizontal=(q[:,None,[0,2]]-center[None,j:j+1024]).abs().sub(half[None,j:j+1024]).clamp_min(0)
                    delta=q[:,None,1]-height[None,j:j+1024]
                    vertical=torch.where(interior[:,None],-delta,delta).clamp_min(0)
                    value,choice=(horizontal.square().sum(-1)+vertical.square()).min(-1)
                    index=torch.where(value<best,choice+j,index);best=torch.minimum(best,value)
            difference=q[:,[0,2]]-center[index]
            horizontal=difference.abs().sub(half[index]).clamp_min(0)*difference.sign()
            sign=torch.where(interior,-torch.ones_like(q[:,1]),torch.ones_like(q[:,1]))
            vertical=((q[:,1]-height[index])*sign).clamp_min(0)
            vector=torch.stack((horizontal[:,0]*sign,vertical,horizontal[:,1]*sign),-1)
            magnitude=vector.norm(dim=-1)
            distances.append(sign*magnitude);normals.append(vector/magnitude.clamp_min(1e-8)[:,None])
            edge=torch.stack((q[:,0]-(origin[0]-self.dx/2),origin[0]+(hf.shape[1]-.5)*self.dx-q[:,0],
                q[:,2]-(origin[2]-self.dx/2),origin[2]+(hf.shape[0]-.5)*self.dx-q[:,2]),-1).amin(-1)
            hc,hh,_=holes
            if len(hc):
                for j in range(0,len(hc),1024):
                    lower=(q[:,None,[0,2]]-hc[None,j:j+1024]).abs().sub(hh[None,j:j+1024]).clamp_min(0).norm(dim=-1).amin(-1)
                    edge=torch.minimum(edge,lower)
            certainty.append(valid[start:start+len(q)] & (magnitude.detach()<=edge.detach()+1e-6))
        valid=torch.cat(certainty).reshape(points.shape[:-1])
        distance=torch.cat(distances).reshape(points.shape[:-1]);normal=torch.cat(normals).reshape(points.shape)
        return torch.where(valid,distance,torch.zeros_like(distance)),torch.where(valid[...,None],normal,torch.zeros_like(normal)),valid


class PackedPrismSDF:
    """Batched active-prism selection with the scalar query's exact geometry."""
    def __init__(self,fields,scenes,device,packed=None):
        from .terrain_batched_geometry import PackedHeightfields
        self.heights=PackedHeightfields(scenes,device,packed=packed)
        sample=torch.zeros(1,3,device=device)
        values=[field._tensors(sample) for field in fields]
        def collect(offset):
            size=max(len(v[offset][0]) for v in values)
            centers=sample.new_zeros(len(fields),size,2);halves=centers.clone()
            levels=sample.new_zeros(len(fields),size);valid=torch.zeros_like(levels,dtype=torch.bool)
            for row,value in enumerate(values):
                c,h,y=value[offset];count=len(c)
                centers[row,:count]=c;halves[row,:count]=h;levels[row,:count]=y;valid[row,:count]=True
            return centers,halves,levels,valid
        self.boxes=collect(0);self.holes=collect(1)
        self.origins=torch.stack([v[4] for v in values]);self.spacing=sample.new_tensor([f.dx for f in fields])
        self.sizes=sample.new_tensor([f.heights.shape for f in fields])

    def query(self,points):
        if points.shape[0]!=len(self.origins) or points.shape[-1]!=3:raise ValueError('Batched XYZ scene mismatch')
        if not torch.isfinite(points).all():raise ValueError('Finite queries required')
        shape=points.shape;points=points.reshape(shape[0],-1,3)
        gap,known=self.heights.query(points);inside=gap.detach()<0
        centers,halves,levels,exists=self.boxes
        values=[];directions=[];certificates=[]
        chunk=max(16,min(512,8_000_000//max(1,len(points)*centers.shape[1])))
        for start in range(0,points.shape[1],chunk):
            p=points[:,start:start+chunk];interior=inside[:,start:start+chunk]
            with torch.no_grad():
                horizontal=(p[:,:,None,[0,2]]-centers[:,None]).abs().sub(halves[:,None]).clamp_min(0)
                delta=p[:,:,None,1]-levels[:,None]
                vertical=torch.where(interior[:,:,None],-delta,delta).clamp_min(0)
                distance=(horizontal.square().sum(-1)+vertical.square()).masked_fill(~exists[:,None],torch.inf)
                index=distance.argmin(-1)
            row=torch.arange(len(p),device=p.device)[:,None]
            difference=p[...,[0,2]]-centers[row,index]
            horizontal=(difference.abs()-halves[row,index]).clamp_min(0)*difference.sign()
            sign=torch.where(interior,-torch.ones_like(p[...,1]),torch.ones_like(p[...,1]))
            vertical=((p[...,1]-levels[row,index])*sign).clamp_min(0)
            vector=torch.stack((horizontal[...,0]*sign,vertical,horizontal[...,1]*sign),-1)
            magnitude=vector.norm(dim=-1)
            values.append(sign*magnitude);directions.append(vector/magnitude.clamp_min(1e-8)[...,None])
            ox,oz=self.origins[:,0,None],self.origins[:,2,None];dx=self.spacing[:,None]
            edge=torch.stack((p[...,0]-(ox-dx/2),ox+(self.sizes[:,1,None]-.5)*dx-p[...,0],
                p[...,2]-(oz-dx/2),oz+(self.sizes[:,0,None]-.5)*dx-p[...,2]),-1).amin(-1)
            hc,hh,_,hv=self.holes
            if hc.shape[1]:
                lower=(p[:,:,None,[0,2]]-hc[:,None]).abs().sub(hh[:,None]).clamp_min(0).norm(dim=-1)
                edge=torch.minimum(edge,lower.masked_fill(~hv[:,None],torch.inf).amin(-1))
            certificates.append(known[:,start:start+len(p[0])] & (magnitude.detach()<=edge.detach()+1e-6))
        valid=torch.cat(certificates,1).reshape(shape[:-1])
        distance=torch.cat(values,1).reshape(shape[:-1]);normal=torch.cat(directions,1).reshape(shape)
        return torch.where(valid,distance,torch.zeros_like(distance)),torch.where(valid[...,None],normal,torch.zeros_like(normal)),valid


@torch.no_grad()
def joint_sdf_features(rep,history,proposal,scenes,*,fields=None,packed_fields=None,max_radius_m=4.,ground_y=None):
    ids=[rep.skeleton.bone_order_names.index(n) for n in JOINT_SDF_NAMES]
    motion=history if proposal is None else torch.cat((history,proposal[:,4:-4]),1)
    joints=rep.inverse(motion,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][:,:,ids]
    root=history[:,-1,:3];yaw=torch.atan2(history[:,-1,4],history[:,-1,3]);c,s=yaw.cos(),yaw.sin()
    values=[];planes=[];masks=[]
    queried=packed_fields.query(joints) if packed_fields is not None else None
    for i,scene in enumerate(scenes):
        if queried is None:
            field=fields[i] if fields is not None else HeightfieldPrismSDF(scene)
            d,n,valid=field.query(joints[i])
        else:d,n,valid=(value[i] for value in queried)
        relative=joints[i]-root[i]
        valid&=relative[...,[0,2]].norm(dim=-1)<=max_radius_m
        position=torch.stack((c[i]*relative[...,0]-s[i]*relative[...,2],relative[...,1],s[i]*relative[...,0]+c[i]*relative[...,2]),-1)
        direction=torch.stack((c[i]*n[...,0]-s[i]*n[...,2],n[...,1],s[i]*n[...,0]+c[i]*n[...,2]),-1)
        time=(torch.arange(joints.shape[1],device=history.device,dtype=history.dtype)-3)/30
        features=torch.cat((torch.asinh(position),torch.asinh(d)[...,None],direction,
            time[:,None,None].expand(-1,15,1)),dim=-1)
        values.append(torch.where(valid[...,None],features,torch.zeros_like(features)));masks.append(valid)
        if ground_y is not None:
            plane_d=joints[i,...,1]-ground_y[i]
            plane_n=torch.zeros_like(direction);plane_n[...,1]=(plane_d.abs()>1e-8).to(plane_n)
            plane=torch.cat((torch.asinh(position),torch.asinh(plane_d)[...,None],plane_n,
                time[:,None,None].expand(-1,15,1)),dim=-1)
            planes.append(torch.where(valid[...,None],plane,torch.zeros_like(plane)))
    return torch.stack(values),torch.stack(masks),torch.stack(planes) if planes else None
