"""Authored heightfield test arena and exact merged top/riser surface geometry.

Coordinates are native SOMA Y-up. Every cell is a constant-height box footprint,
matching the current nearest-cell observation query. No overhangs are implied.
"""

import numpy as np


ZONE_TYPES = [
    ('stairs', f'Stairs {h:.0f} cm', h/100) for h in (8, 12, 16, 20, 24, 30)
] + [
    ('ramp', f'Ramp {a} deg', a) for a in (5, 10, 15, 22)
] + [
    ('cross_slope', f'Cross slope {a} deg', a) for a in (5, 10)
] + [
    ('platform', f'Platforms {h:.1f} m', h) for h in (.4, .8, 1.2, 1.8)
] + [
    ('rocks', f'Rock field {h:.0f} cm', h/100) for h in (6, 12, 20, 35)
] + [
    ('stones', f'Stepping stones {w:.1f} m', w) for w in (.4, .6, .8, 1.)
] + [
    ('trench', f'Gaps {w:.1f} m', w) for w in (.4, .8, 1.2, 1.8)
] + [
    ('slalom', f'Obstacle lanes {h:.1f} m', h) for h in (.3, .6, .9, 1.2)
] + [
    ('mixed', f'Mixed course {i+1}', i) for i in range(4)
]


def build_world(seed=20261011, dx=.2, size=256.):
    if dx != .2 or size != 256.:
        raise ValueError('The authored arena has a fixed256m/.2m contract')
    count = int(round(size/dx))
    origin = np.array([-size/2+dx/2, 0., -size/2+dx/2], dtype=np.float32)
    heights = np.zeros((count, count), dtype=np.float32)
    zones = []
    centers = np.arange(-100, 101, 40)
    for index, (kind, label, value) in enumerate(ZONE_TYPES):
        cx, cz = float(centers[index%6]), float(centers[index//6])
        xids = np.arange(round((cx-15-origin[0])/dx), round((cx+15-origin[0])/dx)+1)
        zids = np.arange(round((cz-15-origin[2])/dx), round((cz+15-origin[2])/dx)+1)
        x, z = np.meshgrid(origin[0]+xids*dx-cx, origin[2]+zids*dx-cz)
        h = np.zeros_like(x, dtype=np.float32)
        rng = np.random.default_rng(seed+index)
        active = (np.abs(x)<11) & (z>-8) & (z<11)
        if kind == 'stairs':
            rise = np.clip(np.floor((z+8)/.6)+1, 0, 10)
            fall = np.clip(np.floor((11-z)/.6)+1, 0, 10)
            stairs = np.minimum(rise, fall)*value
            h[active & (np.abs(x)<3)] = stairs[active & (np.abs(x)<3)]
            # Wide side steps and a narrow flight share each riser height.
            for side, width in ((-7, 2.5), (7, 1.0)):
                mask = active & (np.abs(x-side)<width)
                h[mask] = stairs[mask]
        elif kind == 'ramp':
            length = 7.
            profile = np.minimum(np.clip(z+8,0,length),np.clip(11-z,0,length))*np.tan(np.deg2rad(value))
            mask=active & (np.abs(x)<7);h[mask]=profile[mask]
        elif kind == 'cross_slope':
            taper=np.minimum(np.clip((z+8)/3,0,1),np.clip((11-z)/3,0,1))
            cross=np.clip(x+7,0,14)*np.tan(np.deg2rad(value))*taper
            mask=active & (np.abs(x)<7);h[mask]=cross[mask]
        elif kind == 'platform':
            for px,pz,level in ((-6,-3,value),(3,2,value*1.5),(-4,7,value*2)):
                h[(np.abs(x-px)<3)&(np.abs(z-pz)<2.5)] = level
                approach=(np.abs(x-px)<1.2)&(z>pz-5)&(z<=pz-2.5)
                h[approach]=np.ceil(np.clip((z[approach]-(pz-5))/2.5,0,1)*5)/5*level
        elif kind == 'rocks':
            # Broad finite rock tops retain inspectable ledges rather than visual noise.
            for _ in range(90):
                px,pz=rng.uniform(-10,10),rng.uniform(-6,9)
                radius=rng.uniform(.45,1.6);height=rng.uniform(.2,1)*value
                distance=((x-px)/radius)**2+((z-pz)/(radius*rng.uniform(.7,1.3)))**2
                stone=np.maximum(0,1-distance)*height
                stone=np.round(stone/.025)*.025
                h=np.maximum(h,stone.astype(np.float32))
        elif kind == 'stones':
            h[active]=-.6
            for pz in np.arange(-7.5,10.5,2.):
                for px in np.arange(-9,10,2.):
                    mask=active&(np.abs(x-px)<(2-value)/2)&(np.abs(z-pz)<(2-value)/2)
                    h[mask]=rng.choice([0.,.08,.16])
        elif kind == 'trench':
            for pz in (-5,0,5):
                h[active&(np.abs(z-pz)<value/2)]=-.8
            # An outside bypass always remains on the original plane.
        elif kind == 'slalom':
            for row,pz in enumerate((-5,0,5,9)):
                for px in (-7,0,7):
                    px+=1.5 if row%2 else -1.5
                    h[(np.abs(x-px)<2)&(np.abs(z-pz)<.65)]=value
        elif kind == 'mixed':
            h[active&(z<-2)&(np.abs(x)<6)] = (np.clip(np.floor((z[active&(z<-2)&(np.abs(x)<6)]+8)/.8),0,5)*(.1+.025*value))
            plateau=active&(z>=-2)&(z<2)&(np.abs(x)<6);h[plateau]=.5+.125*value
            gap=active&(z>=2)&(z<2.6+.2*value)&(np.abs(x)<6);h[gap]=-.6
            for px in (-4,0,4):
                h[(np.abs(x-px)<1.3)&(z>4)&(z<6)]=.2+.15*value
            ramp=active&(z>=6)&(np.abs(x)<6);h[ramp]=np.clip((11-z[ramp])/5,0,1)*(.4+.1*value)
        h[~np.isfinite(h)]=0
        # Entrance/exit aprons and connecting paths have the unchanged Y=0 plane.
        h[(z<=-10)|(z>=13)]=0
        heights[np.ix_(zids,xids)]=h
        zones.append({'id':f'zone-{index+1:02d}','label':label,'kind':kind,
            'center_xz':[cx,cz],'bounds_xz':[cx-15,cz-15,cx+15,cz+15],
            'spawn_xz':[cx,cz-12],'heading':0.,'parameter':value})
    spawns=[{'id':'hub','label':'Central plaza','spawn_xz':[0.,0.],'heading':0.,'kind':'flat'}]+zones
    return heights,origin,spawns


def merged_surface(heights, origin, dx):
    """Exactly tile constant-height tops and exposed risers; never interpolate."""
    h=np.asarray(heights,dtype=np.float32)
    if h.ndim!=2 or not np.isfinite(h).all():raise ValueError('Finite2D heightfield required')
    vertices=[];faces=[];colors=[]
    def quad(points,shade=1.):
        first=len(vertices);vertices.extend(points);faces.extend(((first,first+1,first+2),(first,first+2,first+3)))
        y=float(np.mean([p[1] for p in points]))
        base=np.array([.57,.64,.61]) if y>=0 else np.array([.43,.48,.54])
        base=np.clip(base+(max(y,0)*.025),0,1)*shade
        colors.extend([base.tolist()]*4)
    ox,_,oz=origin
    active={}
    def top(key,z0,z1):
        a,b,y=key;x0=ox+(a-.5)*dx;x1=ox+(b-.5)*dx
        near=oz+(z0-.5)*dx;far=oz+(z1-.5)*dx
        quad(((x0,y,near),(x0,y,far),(x1,y,far),(x1,y,near)))
    for z,row in enumerate(h):
        edges=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        current={(int(a),int(b),float(row[a])) for a,b in zip(edges[:-1],edges[1:])}
        for key,start in list(active.items()):
            if key not in current:top(key,start,z);del active[key]
        for key in current:active.setdefault(key,z)
    for key,start in active.items():top(key,start,len(h))
    def walls(grid,along_x):
        for boundary in range(1,grid.shape[0]):
            previous,current=grid[boundary-1],grid[boundary]
            changes=np.r_[True,(previous[1:]!=previous[:-1])|(current[1:]!=current[:-1]),True]
            edges=np.flatnonzero(changes)
            for a,b in zip(edges[:-1],edges[1:]):
                low,high=sorted((float(previous[a]),float(current[a])))
                if low==high:continue
                c=(boundary-.5)*dx
                start=(a-.5)*dx;end=(b-.5)*dx
                if along_x:
                    quad(((ox+start,low,oz+c),(ox+end,low,oz+c),(ox+end,high,oz+c),(ox+start,high,oz+c)),.87)
                else:
                    quad(((ox+c,low,oz+start),(ox+c,high,oz+start),(ox+c,high,oz+end),(ox+c,low,oz+end)),.87)
    walls(h,True);walls(h.T,False)
    return {'vertices':np.asarray(vertices,dtype=np.float32).reshape(-1).tolist(),
            'faces':np.asarray(faces,dtype=np.uint32).reshape(-1).tolist(),
            'colors':np.asarray(colors,dtype=np.float32).reshape(-1).tolist(),
            'semantics':'exact constant-height cell tops and exposed risers'}
