import * as THREE from 'three';
import {OrbitControls} from '/web/soma_lab/vendor/OrbitControls.js';

const query=new URLSearchParams(location.search);
const dataRoot=query.get('data')||'/out/terrain_decoder_pilot_20261010/visual_data_v1/';
const $=id=>document.getElementById(id);
const manifest=await fetch(dataRoot+'manifest.json').then(r=>{if(!r.ok)throw Error('Missing motion export');return r.json();});
const reviewStatus=await fetch(dataRoot+'review-status.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null);
const skin=await fetch(dataRoot+'skin.json').then(r=>r.json());
const cache=new Map();
let clipName=query.get('clip')||(manifest.clips.rest?'rest':Object.keys(manifest.clips)[0]), frame=0, playing=true, last=performance.now(), clock=0, pending=0, synchronizing=false;
const titles=Object.fromEntries(Object.keys(manifest.clips).map(name=>[name,name.replaceAll('_',' ').replaceAll('-',' ')]));
const materials=[];
let renderDirty=true,renderCount=0;
function armLabel(arm){return reviewStatus?.clips?.[clipName]?.arms?.[arm]?.label||manifest.labels[arm];}

function material(color){
  const value=new THREE.MeshStandardMaterial({color,roughness:.8,metalness:0,side:THREE.DoubleSide});
  value.onBeforeCompile=shader=>{
    shader.vertexShader='attribute vec4 skinIndex2;\nattribute vec4 skinWeight2;\n'+shader.vertexShader;
    const v=['x','y','z','w'].map(c=>`skinned += getBoneMatrix(skinIndex2.${c}) * skinVertex * skinWeight2.${c};`).join('\n');
    const n=['x','y','z','w'].map(c=>`skinMatrix += skinWeight2.${c} * getBoneMatrix(skinIndex2.${c});`).join('\n');
    shader.vertexShader=shader.vertexShader.replace('#include <skinning_vertex>',THREE.ShaderChunk.skinning_vertex.replace('transformed =',v+'\ntransformed ='));
    shader.vertexShader=shader.vertexShader.replace('#include <skinnormal_vertex>',THREE.ShaderChunk.skinnormal_vertex.replace('skinMatrix = bindMatrixInverse',n+'\nskinMatrix = bindMatrixInverse'));
  };
  value.customProgramCacheKey=()=> 'official-soma-eight-weight-v1';return value;
}
class SomaMesh extends THREE.SkinnedMesh{
  applyBoneTransform(index,target){
    const base=target.clone().applyMatrix4(this.bindMatrix),point=new THREE.Vector3(),matrix=new THREE.Matrix4();target.set(0,0,0);
    const a=this.geometry.attributes;
    for(let j=0;j<8;j++){
      const offset=index*4+j%4, joint=(j<4?a.skinIndex:a.skinIndex2).array[offset],weight=(j<4?a.skinWeight:a.skinWeight2).array[offset];
      matrix.multiplyMatrices(this.skeleton.bones[joint].matrixWorld,this.skeleton.boneInverses[joint]);
      target.addScaledVector(point.copy(base).applyMatrix4(matrix),weight);
    }return target.applyMatrix4(this.bindMatrixInverse);
  }
}
function human(color){
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(skin.positions,3));g.setIndex(skin.faces);g.computeVertexNormals();
  for(let half=0;half<2;half++){
    const ids=new Uint16Array(skin.positions.length/3*4),weights=new Float32Array(ids.length);
    for(let v=0;v<skin.positions.length/3;v++)for(let j=0;j<4;j++){
      const index=half*4+j;if(index<skin.influences){ids[v*4+j]=skin.indices[v*skin.influences+index];weights[v*4+j]=skin.weights[v*skin.influences+index];}
    }
    g.setAttribute(half?'skinIndex2':'skinIndex',new THREE.Uint16BufferAttribute(ids,4));g.setAttribute(half?'skinWeight2':'skinWeight',new THREE.Float32BufferAttribute(weights,4));
  }
  const bones=[],inverses=[];
  for(let i=0;i<77;i++){const b=new THREE.Bone();b.matrixAutoUpdate=false;b.matrixWorldAutoUpdate=false;const inv=new THREE.Matrix4().fromArray(skin.inverse_bind,i*16).transpose();b.matrixWorld.copy(inv).invert();bones.push(b);inverses.push(inv);}
  const mesh=new SomaMesh(g,material(color));mesh.bind(new THREE.Skeleton(bones,inverses),new THREE.Matrix4());mesh.bindMode=THREE.DetachedBindMode;mesh.frustumCulled=false;return mesh;
}
const panels=[0,1].map(i=>{
  const host=$('view'+i),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0xf0f3f4);renderer.outputColorSpace=THREE.SRGBColorSpace;host.append(renderer.domElement);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.02,500),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
  scene.add(new THREE.HemisphereLight(0xffffff,0x89958e,2.4));const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(3,7,5);scene.add(light);
  const mesh=human(i?0xd68a55:0x397fa1);scene.add(mesh);
  const grid=new THREE.GridHelper(30,30,0x909da2,0xc7d0d3);grid.position.y=.002;scene.add(grid);
  const result={host,renderer,scene,camera,controls,mesh,grid,ground:null,trail:null,arm:query.get(i?'right':'left')||(i?'both':'source'),rig:null};
  controls.addEventListener('change',()=>{renderDirty=true;if(synchronizing)return;synchronizing=true;for(const other of panels){if(other!==result){other.camera.position.copy(camera.position);other.camera.quaternion.copy(camera.quaternion);other.controls.target.copy(controls.target);other.controls.update();}}synchronizing=false;});
  return result;
});
function bounds(){
  const b=new THREE.Box3();for(const arm of Object.values(manifest.clips[clipName].arms))for(const root of arm.roots)b.expandByPoint(new THREE.Vector3(...root));b.min.y=Math.min(0,b.min.y-1.2);b.max.y+=1.2;b.expandByScalar(.7);return b;
}
function ground(scene,b){
  const hf=scene.heightfield_y_m,o=scene.origin,dx=scene.dx,positions=[];
  const bottom=Math.min(b.min.y,-1.5);
  const x0=Math.max(0,Math.floor((b.min.x-o[0])/dx)-2),x1=Math.min(hf[0].length-1,Math.ceil((b.max.x-o[0])/dx)+2);
  const z0=Math.max(0,Math.floor((b.min.z-o[2])/dx)-2),z1=Math.min(hf.length-1,Math.ceil((b.max.z-o[2])/dx)+2);
  function quad(a,b,c,d){positions.push(...a,...b,...c,...a,...c,...d);}
  for(let z=z0;z<=z1;z++)for(let x=x0;x<=x1;x++){
    const y=hf[z][x];if(!Number.isFinite(y))continue;const px=o[0]+x*dx,pz=o[2]+z*dx,a=px-dx/2,c=px+dx/2,d=pz-dx/2,e=pz+dx/2;
    quad([a,y,d],[a,y,e],[c,y,e],[c,y,d]);
    for(const [ix,iz,edge] of [[-1,0,[[a,d],[a,e]]],[1,0,[[c,e],[c,d]]],[0,-1,[[c,d],[a,d]]],[0,1,[[a,e],[c,e]]]]){
      const h=hf[z+iz]?.[x+ix],low=Number.isFinite(h)?Math.max(bottom,h):bottom;if(low>=y)continue;
      quad([edge[0][0],y,edge[0][1]],[edge[1][0],y,edge[1][1]],[edge[1][0],low,edge[1][1]],[edge[0][0],low,edge[0][1]]);
    }
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.computeVertexNormals();
  const m=new THREE.MeshStandardMaterial({color:0xcbd2cf,roughness:.95,side:THREE.DoubleSide});materials.push(m);return new THREE.Mesh(g,m);
}
function resetCamera(){
  renderDirty=true;
  const b=bounds(),center=b.getCenter(new THREE.Vector3()),size=b.getSize(new THREE.Vector3());
  const vertical=panels[0].camera.fov*Math.PI/360,aspect=Math.min(...panels.map(p=>p.camera.aspect));
  const limiting=Math.min(vertical,Math.atan(Math.tan(vertical)*aspect));
  const radius=Math.max(3.8,size.length()*.5/Math.sin(limiting)*1.05);
  synchronizing=true;for(const p of panels){p.controls.target.copy(center);p.camera.position.copy(center).add(new THREE.Vector3(.9,.5,1).normalize().multiplyScalar(radius));p.controls.update();p.grid.position.x=center.x;p.grid.position.z=center.z;}synchronizing=false;
}
async function loadPanel(p){
  const entry=manifest.clips[clipName].arms[p.arm],key=entry.rig;
  p.mesh.visible=false;p.rig=null;
  if(!cache.has(key))cache.set(key,fetch(dataRoot+key,{cache:'no-store'}).then(r=>{if(r.status!==200)throw Error('Missing rig recording');return r.json();}).then(value=>{const text=atob(value.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return new Float32Array(bytes.buffer);}));
  p.rig=await cache.get(key);p.entry=entry;
  if(p.rig.length!==entry.frames*77*16)throw Error(`Rig frame count mismatch: ${key}, ${p.rig.length}, expected ${entry.frames*77*16}`);
  p.mesh.visible=true;
  if(p.trail){p.scene.remove(p.trail);p.trail.geometry.dispose();}
  p.trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(entry.roots.map(r=>new THREE.Vector3(...r))),new THREE.LineBasicMaterial({color:p.arm==='source'?0x286b8c:0xad5b30}));p.scene.add(p.trail);
}
async function loadClip(){
  const version=++pending;playing=false;frame=0;clock=0;
  const clip=manifest.clips[clipName];$('timeline').max=clip.frames-1;
  for(const [i,id] of ['left','right'].entries()){
    const select=$(id),prior=panels[i].arm;select.replaceChildren();for(const [name] of Object.entries(clip.arms)){const o=document.createElement('option');o.value=name;o.textContent=armLabel(name);select.append(o);}
    panels[i].arm=clip.arms[prior]?prior:Object.keys(clip.arms)[i?Object.keys(clip.arms).length-1:0];select.value=panels[i].arm;
    if(panels[i].ground){panels[i].scene.remove(panels[i].ground);panels[i].ground.geometry.dispose();panels[i].ground.material.dispose();}
    panels[i].ground=ground(clip.scene,bounds());panels[i].scene.add(panels[i].ground);
  }
  await Promise.all(panels.map(loadPanel));if(version!==pending)return;resetCamera();applyTransparency();setFrame(0);playing=true;last=performance.now();$('status').textContent=manifest.trained_model_output===false?'Source motion recordings | 30 Hz':'Recorded motion | 30 Hz';
}
function applyTransparency(){renderDirty=true;for(const p of panels){if(!p.ground)continue;const m=p.ground.material;m.transparent=$('transparent').checked;m.opacity=m.transparent ? .25 : 1;m.depthWrite=!m.transparent;m.needsUpdate=true;}}
function activeFrames(){return Math.max(1,...panels.map(p=>manifest.clips[clipName].arms[p.arm]?.frames||0));}
function setFrame(value){
  renderDirty=true;
  const total=activeFrames();$('timeline').max=total-1;
  frame=Math.max(0,Math.min(total-1,Math.floor(value)));$('timeline').value=frame;$('time').textContent=`${((frame+1)/30).toFixed(2)} s | ${frame+1}/${total}`;
  for(let i=0;i<2;i++){
    const p=panels[i];if(!p.rig)continue;const shown=Math.min(frame,p.entry.frames-1);for(let j=0;j<77;j++)p.mesh.skeleton.bones[j].matrixWorld.fromArray(p.rig,(shown*77+j)*16).transpose();p.mesh.skeleton.update();
    p.trail.geometry.setDrawRange(0,shown+1);p.trail.visible=$('trail').checked;
    $('label'+i).textContent=armLabel(p.arm)+`\n${((shown+1)/30).toFixed(2)} / ${(p.entry.frames/30).toFixed(2)} s | ${shown+1}/${p.entry.frames}`+(frame>=p.entry.frames?' | Ended':'')+`\nRoot Y: ${p.entry.roots[shown][1].toFixed(3)} m`;
  }
}
function resize(){renderDirty=true;for(const p of panels){const r=p.host.getBoundingClientRect();p.renderer.setSize(r.width,r.height,false);p.camera.aspect=r.width/r.height;p.camera.updateProjectionMatrix();}}
for(const [name,title] of Object.entries(titles)){const o=document.createElement('option');o.value=name;o.textContent=title;$('clip').append(o);}$('clip').value=clipName;
$('clip').onchange=()=>{clipName=$('clip').value;loadClip();};
for(const [i,id] of ['left','right'].entries())$(id).onchange=async()=>{playing=false;panels[i].arm=$(id).value;await loadPanel(panels[i]);setFrame(frame);};
$('play').onclick=()=>{if(!playing&&frame===activeFrames()-1)setFrame(0);playing=!playing;clock=0;last=performance.now();};$('previous').onclick=()=>{playing=false;setFrame(frame-1);};$('next').onclick=()=>{playing=false;setFrame(frame+1);};$('timeline').oninput=()=>{playing=false;setFrame(Number($('timeline').value));};$('transparent').onchange=applyTransparency;$('trail').onchange=()=>setFrame(frame);$('reset').onclick=resetCamera;
new ResizeObserver(resize).observe(document.querySelector('main'));lucide.createIcons();resize();await loadClip();
window.pilotView={manifest,panels,setFrame,selectClip:async name=>{clipName=name;$('clip').value=name;await loadClip();playing=false;},pause:()=>playing=false,get frame(){return frame;},get playing(){return playing;}};
function animate(now){const delta=Math.min((now-last)/1000,.1);last=now;if(playing){clock+=delta*30*Number($('speed').value);if(clock>=1){const step=Math.floor(clock);clock-=step;const next=frame+step;setFrame(next);if(next>=activeFrames()-1){playing=false;clock=0;}}}if(renderDirty){for(const p of panels)p.renderer.render(p.scene,p.camera);renderDirty=false;document.body.dataset.renderCount=String(++renderCount);}requestAnimationFrame(animate);}requestAnimationFrame(animate);
