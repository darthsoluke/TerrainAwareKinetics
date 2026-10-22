import {SCHEMA,CHANNEL,STORAGE,defaults,normalizeSettings,preset,runtimeEqual} from '/tuning-contract.js';

const $=id=>document.getElementById(id);
const channel=new BroadcastChannel(CHANNEL);
const viewers=new Map();
const embedded=new URLSearchParams(location.search).get('embedded')==='1';
const pinnedViewer=embedded?new URLSearchParams(location.search).get('viewer'):null;
let contract,draft,selected=null,dirty=false,latest=null;
const labels={refresh_frames:'重规划间隔',num_tokens:'预测编码数',inference_steps:'生成轮数',
  lookahead_m:'前视距离',lateral_halfwidth_m:'观察半宽',lookahead_gain:'速度前视系数',lookahead_base_m:'基础前视距离'};
const icons=()=>window.lucide?.createIcons({attrs:{'stroke-width':1.7}});
const say=(text,error=false)=>{$('message').textContent=text;$('message').classList.toggle('error',error);};
function format(key,value){
  if(value===undefined)return '—';
  if(key==='refresh_frames')return `${value} 帧`;
  if(key==='lookahead_m'&&value<0)return '自动';
  if(key.endsWith('_m'))return `${Number(value).toFixed(1)} m`;
  return String(value);
}
function renderDraft(){
  $('horizon-controls').hidden=!contract.accelerated_horizons;
  $('horizon-label').textContent=contract.planning_policy?'前瞻Root · 近期执行8':'完整规划';
  const selectedHorizon=draft.planning_policy?.lookahead_tokens??draft.runtime.num_tokens;
  for(const input of document.querySelectorAll('[data-horizon]'))input.checked=Number(input.dataset.horizon)===selectedHorizon;
  for(const id of ['num_tokens','inference_steps','refresh_frames','preset']){
    $(id).disabled=Boolean(contract.planning_policy);
    const twin=document.querySelector(`[data-number-for="${id}"]`);if(twin)twin.disabled=Boolean(contract.planning_policy);
  }
  $('speed-governor-controls').hidden=!contract.speed_governor;
  if(draft.speed_governor)for(const input of document.querySelectorAll('[data-governor]')){
    if(input.type==='checkbox')input.checked=draft.speed_governor[input.dataset.governor];
    else input.value=draft.speed_governor[input.dataset.governor];
  }
  for(const input of document.querySelectorAll('[data-runtime]')){
    const n=draft.runtime[input.id];input.value=input.id==='lookahead_m'&&n<0?2.8:n;
  }
  for(const input of document.querySelectorAll('[data-extra]'))input.value=draft[input.id];
  $('auto_lookahead').checked=draft.runtime.lookahead_m<0;
  $('invert_y').checked=draft.invert_y;
  for(const input of document.querySelectorAll('[data-number-for]'))input.value=$(input.dataset.numberFor).value;
  const known={24:'original',8:'responsive',4:'sensitive',1:'per_frame'};
  $('preset').value=Object.keys(contract.defaults).every(key=>key==='refresh_frames'||draft.runtime[key]===contract.defaults[key])
    ? known[draft.runtime.refresh_frames]??'custom':'custom';
  updatePreview();
}
function readDraft(){
  if(draft.speed_governor)for(const input of document.querySelectorAll('[data-governor]'))
    draft.speed_governor[input.dataset.governor]=input.type==='checkbox'?input.checked:Number(input.value);
  for(const input of document.querySelectorAll('[data-runtime]'))draft.runtime[input.id]=Number(input.value);
  if($('auto_lookahead').checked)draft.runtime.lookahead_m=-1;
  for(const input of document.querySelectorAll('[data-extra]'))draft[input.id]=Number(input.value);
  draft.invert_y=$('invert_y').checked;
  dirty=true;$('preset').value='custom';updatePreview();
}
function updatePreview(){
  const r=draft.runtime,auto=r.lookahead_m<0;
  for(const id of ['lookahead_m','lookahead_gain','lookahead_base_m']){
    $(id).disabled=id==='lookahead_m'?auto:!auto;
    document.querySelector(`[data-number-for="${id}"]`).disabled=$(id).disabled;
  }
  const horizon=(4*r.num_tokens-4)/30;
  const extent=auto?Math.min(20,Math.max(.8,draft.speed_mps*horizon*r.lookahead_gain+r.lookahead_base_m)):r.lookahead_m;
  $('response-frequency').textContent=`目标 ${(30/r.refresh_frames).toFixed(2)} Hz`;
  $('lookahead-preview').textContent=`当前指令约 ${extent.toFixed(2)} m`;
  $('horizon-preview').textContent=`${4*r.num_tokens} 帧窗口 / ${4*r.num_tokens-8} 帧可播放`;
  $('calls-per-packet').textContent=r.refresh_frames<8?`${8/r.refresh_frames} 次 / 8 帧`:`每 ${r.refresh_frames} 帧约 1 次`;
  const d=contract.defaults;
  const fast=contract.accelerated_horizons;
  const accelerated=fast ? fast.tokens.includes(r.num_tokens)&&r.inference_steps===fast.inference_steps&&r.path_refinements===0
    :Object.keys(d).filter(k=>k!=='refresh_frames').every(k=>r[k]===d[k]);
  $('execution-path').textContent=accelerated?'CUDA 加速':'原生实验 · 未加速';
  const active=latest?.applied_runtime;
  $('applied-horizon').textContent=active?(contract.planning_policy
    ?`前瞻 ${latest.applied_policy?.lookahead_tokens??'—'} · 近期Root/Pose ${active.num_tokens} · CUDA 加速`
    :`Root ${latest.root_prediction_frames??0} 帧 · ${((4*active.num_tokens-8)/30).toFixed(2)} s · ${latest.neural_execution==='split-cuda-graphs'?'CUDA 加速':'原生执行'}`):'等待动作';
  renderComparison();
  try{normalizeSettings(draft,contract);$('apply').disabled=!latest;}
  catch(error){$('apply').disabled=true;say(error.message,true);}
}
function renderComparison(){
  const applied=latest?.applied_runtime;
  $('comparison-rows').replaceChildren();
  for(const [key,label] of Object.entries(labels)){
    const tr=document.createElement('tr');
    for(const text of [label,format(key,draft.runtime[key]),format(key,applied?.[key])]){
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    }
    if(applied?.[key]!==draft.runtime[key])tr.children[1].className='pending';
    $('comparison-rows').append(tr);
  }
  const governorEqual=!contract.speed_governor || Object.keys(contract.speed_governor.defaults)
    .every(key=>draft.speed_governor[key]===latest?.applied_governor?.[key]);
  const policyEqual=!contract.planning_policy || draft.planning_policy.lookahead_tokens===latest?.applied_policy?.lookahead_tokens;
  $('apply-state').textContent=!latest?'等待连接':dirty?'有未应用修改':runtimeEqual(draft.runtime,applied)&&governorEqual&&policyEqual?'已生效':'等待动作边界';
}
function apply(){
  try{
    draft=normalizeSettings(draft,contract);
    if(!latest)throw new Error('场景尚未连接');
    channel.postMessage({type:'apply',target:selected,settings:draft});
    dirty=false;say('已发送参数');renderComparison();
  }catch(error){say(error.message,true);}
}
function renderViewers(){
  for(const [id,row] of viewers)if(Date.now()-row.received>3500)viewers.delete(id);
  if(pinnedViewer)selected=pinnedViewer;
  else if(!viewers.has(selected))selected=viewers.keys().next().value??null;
  const previous=latest;latest=selected?viewers.get(selected):null;
  $('viewer').replaceChildren();
  for(const [id,row] of viewers){const option=new Option(`PFNN ${row.map_id} · ${id.slice(0,4)}`,id);$('viewer').append(option);}
  $('viewer').value=selected??'';
  $('connection-dot').classList.toggle('online',Boolean(latest));
  $('connection-text').textContent=latest?'场景已连接':'等待场景连接';
  $('pause').disabled=$('reset-character').disabled=!latest;
  if(latest){
    const governor=latest.governor;
    const states={off:'已关闭',clear:'正常',stopped:'静止',braking:'提前减速',limited:'地形限速',recovering:'恢复速度',boundary:'地图边界',horizon_limited:'视距限速'};
    $('governor-status').textContent=states[governor?.mode]??'等待动作';
    for(const [id,value] of [['governor-requested',governor?.requested_speed_mps],
      ['governor-target',governor?.target_speed_mps],['governor-actual',latest.actual_speed_mps]])
      $(id).textContent=Number.isFinite(value)?`${value.toFixed(2)} m/s`:'—';
    if(!dirty){
      const incoming=normalizeSettings(latest.settings,contract);
      if(JSON.stringify(draft)!==JSON.stringify(incoming)){draft=incoming;renderDraft();}
    }
    $('metric-map').textContent=latest.map_id;
    for(const [id,value,digits] of [['metric-fps',latest.fps,1],['metric-inference',latest.inference_ms,1],
      ['metric-age',latest.observation_age_ms,0],['metric-underruns',latest.underruns,0]])
      $(id).textContent=Number.isFinite(value)?value.toFixed(digits):'—';
    $('pause').querySelector('span').textContent=latest.paused?'继续播放':'暂停';
  }
  if(contract)updatePreview();
}
function savedPresets(){try{return JSON.parse(localStorage.getItem('soma-pfnn-presets-v1')??'{}');}catch{return {};}}
function renderSaved(){
  $('saved-preset').replaceChildren(new Option('选择预设',''));
  for(const name of Object.keys(savedPresets()))$('saved-preset').append(new Option(name,name));
}
channel.onmessage=event=>{
  const data=event.data;
  if(!data||!contract)return;
  if(data.type==='viewer-state'){
    if(pinnedViewer && data.viewer_id!==pinnedViewer)return;
    viewers.set(data.viewer_id,{...data,received:Date.now()});renderViewers();
  }else if(data.type==='applied'&&data.viewer_id===selected){
    draft=normalizeSettings(data.settings,contract);dirty=false;renderDraft();say('参数已接收，模型参数在动作边界生效');
  }else if(data.type==='horizon-applied'&&data.viewer_id===selected){
    const incoming=normalizeSettings(data.settings,contract);
    if(dirty){
      for(const key of ['num_tokens','refresh_frames','inference_steps'])draft.runtime[key]=incoming.runtime[key];
      if(incoming.planning_policy)draft.planning_policy={...incoming.planning_policy};
    }else draft=incoming;
    renderDraft();say('前瞻档位已接收');
  }else if(data.type==='error'&&data.viewer_id===selected)say(data.message,true);
};
async function initialize(){
  if(embedded)initializeEmbeddedLayout();
  const response=await fetch('/api/tuning',{cache:'no-store'});
  if(!response.ok)throw new Error('当前服务不支持调参');
  const data=await response.json();contract=data.runtime;draft=defaults(contract);
  $('model-id').textContent=`模型 ${data.checkpoint_sha256}`;
  try{const saved=localStorage.getItem(STORAGE);if(saved)draft=normalizeSettings(JSON.parse(saved),contract);}catch{say('已忽略不兼容的本地参数',true);}
  renderDraft();renderSaved();icons();
  for(const input of document.querySelectorAll('[data-horizon]'))input.addEventListener('change',()=>{
    if(!input.checked)return;
    channel.postMessage({type:'apply-horizon',target:selected,tokens:Number(input.dataset.horizon)});
    say('已发送前瞻档位');
  });
  for(const input of document.querySelectorAll('[data-runtime],[data-extra],[data-governor]'))input.addEventListener('input',()=>{
    const twin=document.querySelector(`[data-number-for="${input.id}"]`);if(twin)twin.value=input.value;readDraft();
    if(input.id==='governor-enabled')apply();
  });
  for(const input of document.querySelectorAll('[data-number-for]'))input.addEventListener('input',()=>{
    if(input.value===''||!input.checkValidity()){say('请输入范围内的数值',true);$('apply').disabled=true;return;}
    $(input.dataset.numberFor).value=input.value;readDraft();
  });
  $('auto_lookahead').onchange=$('invert_y').onchange=readDraft;
  $('preset').onchange=()=>{if($('preset').value==='custom')return;draft=preset($('preset').value,contract,draft);dirty=true;renderDraft();say('预设已载入，尚未应用');};
  $('apply').onclick=apply;
  $('restore').onclick=()=>{draft=defaults(contract);dirty=true;$('preset').value='original';renderDraft();apply();};
  $('viewer').onchange=()=>{selected=$('viewer').value;latest=null;dirty=false;renderViewers();};
  $('pause').onclick=()=>channel.postMessage({type:'toggle-pause',target:selected});
  $('reset-character').onclick=()=>channel.postMessage({type:'reset-character',target:selected});
  $('parameters').onsubmit=event=>{event.preventDefault();apply();};
  $('save-preset').onclick=()=>{try{const name=$('preset-name').value.trim();if(!name)throw new Error('请输入预设名称');const all=savedPresets();all[name]=normalizeSettings(draft,contract);localStorage.setItem('soma-pfnn-presets-v1',JSON.stringify(all));renderSaved();$('saved-preset').value=name;say('预设已保存');}catch(error){say(error.message,true);}};
  $('saved-preset').onchange=()=>{const value=savedPresets()[$('saved-preset').value];if(value){draft=normalizeSettings(value,contract);dirty=true;$('preset').value='custom';renderDraft();say('预设已载入，尚未应用');}};
  $('delete-preset').onclick=()=>{const name=$('saved-preset').value;if(!name)return;const all=savedPresets();delete all[name];localStorage.setItem('soma-pfnn-presets-v1',JSON.stringify(all));renderSaved();say('预设已删除');};
  $('export').onclick=()=>{try{const value=normalizeSettings(draft,contract);const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='motionbricks-settings.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){say(error.message,true);}};
  $('import').onclick=()=>$('import-file').click();
  $('import-file').onchange=async()=>{try{const file=$('import-file').files[0];if(!file)return;if(file.size>1048576)throw new Error('参数文件过大');draft=normalizeSettings(JSON.parse(await file.text()),contract);dirty=true;$('preset').value='custom';renderDraft();say('参数已导入，尚未应用');}catch(error){say(error.message,true);}finally{$('import-file').value='';}};
  channel.postMessage({type:'hello'});
  setInterval(()=>{renderViewers();channel.postMessage({type:'hello'});},1000);
}

function initializeEmbeddedLayout(){
  document.documentElement.classList.add('tuning-embedded');
  const nav=document.querySelector('.section-nav');
  nav.setAttribute('role','tablist');
  const sections=[...document.querySelectorAll('.parameter-section')];
  const names={response:'响应',terrain:'地形',generation:'生成',view:'镜头',fixed:'约定'};
  const links=[...nav.querySelectorAll('a')];
  const activate=id=>{
    for(const section of sections)section.hidden=section.id!==id;
    for(const link of links){
      const active=link.hash===`#${id}`;
      link.setAttribute('aria-selected',String(active));link.tabIndex=active?0:-1;
    }
  };
  for(const [index,link] of links.entries()){
    const id=link.hash.slice(1);link.textContent=names[id];
    link.setAttribute('role','tab');link.setAttribute('aria-controls',id);link.id=`tab-${id}`;
    $(id).setAttribute('role','tabpanel');$(id).setAttribute('aria-labelledby',link.id);
    link.onclick=event=>{event.preventDefault();activate(id);};
    link.onkeydown=event=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
      event.preventDefault();
      const next=event.key==='Home'?0:event.key==='End'?links.length-1:
        (index+(event.key==='ArrowRight'?1:-1)+links.length)%links.length;
      activate(links[next].hash.slice(1));links[next].focus();
    };
  }
  activate('response');
  const table=document.querySelector('.comparison table');
  const actual=document.createElement('details');actual.className='embedded-details';
  const actualSummary=document.createElement('summary');actualSummary.textContent='实际生效参数';
  table.before(actual);actual.append(actualSummary,table);
  const saved=document.createElement('details');saved.className='embedded-details';
  const savedSummary=document.createElement('summary');savedSummary.textContent='个人预设与文件';
  const presets=document.querySelector('.saved-presets');presets.before(saved);
  saved.append(savedSummary,presets,document.querySelector('.file-actions'));
}
initialize().catch(error=>say(error.message,true));
