"""Losses on the actual FK/skinned motion, independent of predicted contact masks."""

from collections import OrderedDict

import numpy as np
import torch

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.data.soma_skin_reference import load_official_skinner,move_skinner
from motionbricks.data.terrain_response_loss import heightfield_gap,source_support_masks
from motionbricks.data.terrain_distance_volume import TerrainDistanceVolume,swept_surface_points
from motionbricks.data.terrain_control import SoftCommandState


class JointGeometryObjective:
    def __init__(self,representation,device,max_volumes=96):
        self.rep=representation
        self.skin=move_skinner(load_official_skinner(representation.skeleton),torch.device(device))
        self.skin.skeleton_skin.float().to(device)
        surfaces,*_=official_sole_surfaces(representation.skeleton)
        soles=np.concatenate([sole_region_indices(surfaces,side) for side in ('Left','Right')])
        dominant=self.skin.lbs_indices.gather(1,self.skin.lbs_weights.argmax(1)[:,None]).squeeze(1).cpu().numpy()
        body=[]
        for joint in np.unique(dominant):
            candidates=np.flatnonzero(dominant==joint)
            body.extend(candidates[np.linspace(0,len(candidates)-1,min(6,len(candidates))).astype(int)].tolist())
        self.indices=np.concatenate((soles,np.asarray(sorted(set(body)-set(soles.tolist())),dtype=np.int64)))
        self.sole_count=len(soles);self.volumes=OrderedDict();self.max_volumes=max_volumes

    def geometry(self,world):
        pose=self.rep.inverse(world.float(),False,return_all=True,joint_positions_from='global_rot_data')
        b,t=world.shape[:2]
        y,xz=region_geometry_from_motion(pose['local_joint_rots'].reshape(b*t,30,3,3),
            world[...,:3].reshape(b*t,3).float(),self.skin,self.indices)
        points=torch.stack((xz[...,0],y,xz[...,1]),-1).reshape(b,t,len(self.indices),3)
        return pose['posed_joints'],points

    @torch.no_grad()
    def prepare_episode(self,episode):
        from motionbricks.data.terrain_batched_geometry import PackedHeightfields,PackedDistanceVolumes
        joints,points=self.geometry(episode.physical_source)
        volumes=[self.volume(scene,episode.physical_source[row:row+1],episode.identities[row])
            for row,scene in enumerate(episode.scenes)]
        return {'joints':joints,'points':points,'volumes':PackedDistanceVolumes(volumes,points.device),
            'heights':PackedHeightfields(episode.scenes,points.device,packed=episode.packed_scene)}

    def prepared_losses(self,prediction,history,prepared,*,offset,diagnostics=True):
        from motionbricks.data.terrain_batched_geometry import batch_swept_surface_points
        frames=prediction.shape[1];last=frames-4
        pj,pp=self.geometry(prediction)
        sj=prepared['joints'][:,offset:offset+frames]
        sp=prepared['points'][:,offset:offset+frames]
        with torch.no_grad():
            hj=self.rep.inverse(history.float(),False,return_all=True,joint_positions_from='global_rot_data')['posed_joints']
        fk=(pj[:,4:last]-sj[:,4:last]).norm(dim=-1).mean()
        boundary=((pj[:,4]-hj[:,-1])-(sj[:,4]-sj[:,3])).norm(dim=-1).mean()*30
        pred=pp[:,4:last];truth=sp[:,4:last]
        swept=batch_swept_surface_points(pred[:,:,::3],4)
        distance,known=prepared['volumes'].query(swept)
        fallback,assessed=prepared['heights'].query(swept)
        distance=torch.where(known,distance,fallback)
        excess=(-distance-.005).clamp_min(0)
        collision=((excess.square()*assessed).flatten(1).sum(1)/assessed.flatten(1).sum(1).clamp_min(1)).mean()
        sole=truth[:,:,:self.sole_count]
        with torch.no_grad():
            gap,valid=prepared['heights'].query(sole)
            active,intervals=source_support_masks(sole,gap,valid)
        error=pred[:,:,:self.sole_count,1]-sole[...,1]
        contact=((error.abs()*active).flatten(1).sum(1)/active.flatten(1).sum(1).clamp_min(1)).mean()
        pv=(pred[:,1:,:self.sole_count]-pred[:,:-1,:self.sole_count])*30
        sv=(sole[:,1:]-sole[:,:-1])*30
        slip=(((pv-sv).norm(dim=-1)*intervals).flatten(1).sum(1)/intervals.flatten(1).sum(1).clamp_min(1)).mean()
        counts={'known_surface_points':int(assessed.sum()),'unknown_surface_points':int((~assessed).sum()),
            'support_vertices':int(active.sum()),'source_collision_p99_m':0.}
        if diagnostics:
            with torch.no_grad():
                swept_source=batch_swept_surface_points(truth[:,:,::3],4)
                sd,sk=prepared['volumes'].query(swept_source);sf,sfk=prepared['heights'].query(swept_source)
                sd=torch.where(sk,sd,sf)
                for row in range(len(sd)):
                    if sfk[row].any():counts['source_collision_p99_m']=max(counts['source_collision_p99_m'],
                        float(torch.quantile((-sd[row][sfk[row]]).clamp_min(0),.99)))
        return {'fk_m':fk,'boundary_mps':boundary,'collision_m2':collision,
            'support_gap_m':contact,'source_support_velocity_mps':slip,**counts}

    @torch.no_grad()
    def full_surface_metrics(self,world,scene,source,identity):
        """Audit every official skinned vertex and eight swept samples per interval."""
        if len(world)!=1:raise ValueError('Full-surface audits require one sequence at a time')
        volume=self.volume(scene,source,identity)
        distances=[];known_count=0;total=0;previous=None
        for start in range(0,world.shape[1],16):
            part=world[0,start:start+16]
            pose=self.rep.inverse(part[None].float(),False,return_all=True,joint_positions_from='global_rot_data')
            points=self.skin.skin(pose['local_joint_rots'][0],part[:,:3,None].transpose(1,2).expand(-1,30,-1),rot_is_global=False)
            if previous is not None:points=torch.cat((previous,points),0)
            previous=points[-1:]
            swept=swept_surface_points(points,8)
            if start+16<world.shape[1]:swept=swept[:-1]
            distance,valid=volume.query(swept)
            fallback,known=heightfield_gap(swept,scene)
            distance=torch.where(valid,distance,fallback)
            distances.append((-distance[known]).clamp_min(0).cpu().numpy())
            total+=known.numel();known_count+=int(known.sum())
        penetration=np.concatenate(distances)
        return {'body_penetration_p99_m':float(np.quantile(penetration,.99)) if len(penetration) else None,
            'body_penetration_max_m':float(penetration.max()) if len(penetration) else None,
            'unknown_body_fraction':1-known_count/max(1,total),'audited_vertices_per_frame':len(self.skin.bind_vertices),
            'swept_substeps':8}

    def volume(self,scene,source,identity):
        hf=torch.as_tensor(scene['heightfield_y_m']).detach().cpu()
        if torch.isfinite(hf).all() and hf.min()==hf.max():
            key=('plane',identity)
            crop=scene
        else:
            origin=torch.as_tensor(scene['min_point_world_xyz_m']).detach().cpu()
            dx=float(scene['dx_m'])
            roots=source.detach().cpu()[...,[0,2]].reshape(-1,2)
            lo=torch.floor((roots.amin(0)-3-origin[[0,2]])/dx).long()
            hi=torch.ceil((roots.amax(0)+3-origin[[0,2]])/dx).long()
            x0=max(0,int(lo[0]));z0=max(0,int(lo[1]))
            x1=min(hf.shape[1],int(hi[0])+1);z1=min(hf.shape[0],int(hi[1])+1)
            key=(identity,x0,z0,x1,z1)
            crop={'heightfield_y_m':hf[z0:z1,x0:x1],
                  'min_point_world_xyz_m':origin+origin.new_tensor([x0*dx,0,z0*dx]),'dx_m':dx}
            if 'heightfield_valid' in scene:crop['heightfield_valid']=torch.as_tensor(scene['heightfield_valid'])[z0:z1,x0:x1]
        if key not in self.volumes:
            self.volumes[key]=TerrainDistanceVolume(crop)
            if len(self.volumes)>self.max_volumes:self.volumes.popitem(last=False)
        self.volumes.move_to_end(key)
        return self.volumes[key]

    @torch.no_grad()
    def barrier_clearance(self,world,row):
        normal=world.new_tensor(row['wall_normal_world_xz']);parts=[]
        for start in range(0,world.shape[1],16):
            part=world[0,start:start+16]
            pose=self.rep.inverse(part[None],False,return_all=True,joint_positions_from='global_rot_data')
            vertices=self.skin.skin(pose['local_joint_rots'][0],part[:,:3,None].transpose(1,2).expand(-1,30,-1),rot_is_global=False)
            parts.append(row['wall_face_projection_m']-(vertices[...,[0,2]]*normal).sum(-1).amax(-1))
        return torch.cat(parts).cpu().numpy()

    def losses(self,prediction,source,history,scenes,identities,*,volume_source=None):
        frames=prediction.shape[1]
        pj,pp=self.geometry(prediction)
        with torch.no_grad():
            sj,sp=self.geometry(source[:,:frames])
            hj,_=self.geometry(history)
        last=frames-4
        fk=(pj[:,4:last]-sj[:,4:last]).norm(dim=-1).mean()
        boundary=((pj[:,4]-hj[:,-1])-(sj[:,4]-sj[:,3])).norm(dim=-1).mean()*30
        collision=pp.sum()*0.;contact=collision;slip=collision
        counts={'known_surface_points':0,'unknown_surface_points':0,'support_vertices':0,
                'source_collision_p99_m':0.}
        for row,scene in enumerate(scenes):
            bounds=source if volume_source is None else volume_source
            volume=self.volume(scene,bounds[row:row+1],identities[row])
            pred=pp[row,4:last];truth=sp[row,4:last]
            swept=swept_surface_points(pred[:,::3],4)
            swept_source=swept_surface_points(truth[:,::3],4)
            distance,known=volume.query(swept)
            source_distance,source_known=volume.query(swept_source)
            # A local distance volume is an acceleration structure. Evaluate
            # its exterior against the actual full scene instead of dropping it.
            fallback,full_known=heightfield_gap(swept,scene)
            fallback_source,full_source_known=heightfield_gap(swept_source,scene)
            distance=torch.where(known,distance,fallback)
            source_distance=torch.where(source_known,source_distance,fallback_source)
            assessed=full_known
            # Known terrain is authoritative. Source skin residuals are reported,
            # rather than becoming per-vertex permission to penetrate the scene.
            excess=(-distance-.005).clamp_min(0)
            collision=collision+(excess.square()*assessed).sum()/assessed.sum().clamp_min(1)
            counts['known_surface_points']+=int(assessed.sum());counts['unknown_surface_points']+=int((~assessed).sum())
            if full_source_known.any():
                counts['source_collision_p99_m']=max(counts['source_collision_p99_m'],
                    float(torch.quantile((-source_distance[full_source_known]).clamp_min(0),.99)))
            gap,valid=heightfield_gap(truth[:,:self.sole_count],scene)
            active,intervals=source_support_masks(truth[:,:self.sole_count],gap,valid)
            # Preserve the actual source support surface correspondence. Looking
            # up terrain below a misplaced prediction would pull a foot onto a
            # wall top instead of learning to avoid that wall.
            height_error=pred[:,:self.sole_count,1]-truth[:,:self.sole_count,1]
            contact=contact+(height_error.abs()*active).sum()/active.sum().clamp_min(1)
            pred_velocity=(pred[1:,:self.sole_count]-pred[:-1,:self.sole_count])*30
            source_velocity=(truth[1:,:self.sole_count]-truth[:-1,:self.sole_count])*30
            slip=slip+((pred_velocity-source_velocity).norm(dim=-1)*intervals).sum()/intervals.sum().clamp_min(1)
            counts['support_vertices']+=int(active.sum())
        b=len(prediction)
        return {'fk_m':fk,'boundary_mps':boundary,'collision_m2':collision/b,
                'support_gap_m':contact/b,'source_support_velocity_mps':slip/b,**counts}
