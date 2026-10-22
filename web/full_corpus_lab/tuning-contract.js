export const SCHEMA='motionbricks.pfnn_runtime_tuning.v1';
export const CHANNEL='soma-pfnn-tuning-v1';
export const STORAGE='soma-pfnn-settings-v1';
export const EXTRA_DEFAULTS={speed_mps:1.6,run_multiplier:1.8,mouse_sensitivity:1,field_of_view:50,invert_y:false};
export function defaults(contract){return {schema:SCHEMA,runtime:{...contract.defaults},...EXTRA_DEFAULTS,
  ...(contract.planning_policy?{planning_policy:{...contract.planning_policy.defaults}}:{}),
  ...(contract.speed_governor?{speed_governor:{...contract.speed_governor.defaults}}:{})};}
export function normalizeSettings(value,contract){
  if(!value || value.schema!==SCHEMA || !value.runtime)throw new Error('参数文件版本不匹配');
  const result={...EXTRA_DEFAULTS,...value,runtime:{...value.runtime}};
  if(contract.planning_policy){
    const declared=contract.planning_policy;
    result.planning_policy={...(value.planning_policy??{...declared.defaults,lookahead_tokens:value.runtime.num_tokens})};
    if(!value.planning_policy){result.runtime.num_tokens=8;result.runtime.refresh_frames=4;result.runtime.inference_steps=4;}
    const p=result.planning_policy;
    if(p.schema!==declared.defaults.schema||p.execute_tokens!==8||!declared.lookahead_choices.includes(p.lookahead_tokens)
      ||Object.keys(p).sort().join()!==Object.keys(declared.defaults).sort().join())throw new Error('前瞻与近期执行协议不匹配');
  }else delete result.planning_policy;
  const keys=Object.keys(contract.defaults).sort();
  if(JSON.stringify(Object.keys(value.runtime).sort())!==JSON.stringify(keys))throw new Error('模型参数不完整');
  const number=(key,low,high,integer=false,source=result)=>{
    const n=source[key];
    if(typeof n!=='number'||!Number.isFinite(n)||n<low||n>high||(integer&&!Number.isInteger(n)))
      throw new Error(`${key} 超出范围 ${low}–${high}`);
  };
  for(const [key,range] of Object.entries(contract.limits)){
    if(key==='refresh_frames'){
      if(!range.includes(result.runtime[key]))throw new Error('不支持这个重规划间隔');
    }else if(!(key==='lookahead_m'&&result.runtime[key]===-1))
      number(key,...range,['num_tokens','inference_steps'].includes(key),result.runtime);
  }
  if(result.runtime.path_refinements!==0)throw new Error('路径修正固定为零');
  if(result.runtime.refresh_frames>4*result.runtime.num_tokens-8)throw new Error('重规划间隔超过可用动作长度');
  number('speed_mps',.3,contract.max_speed_mps);number('run_multiplier',1,3);
  number('mouse_sensitivity',.2,3);number('field_of_view',30,90);
  if(typeof result.invert_y!=='boolean')throw new Error('镜头反转设置无效');
  if(contract.speed_governor){
    result.speed_governor={...(value.speed_governor??contract.speed_governor.defaults)};
    if(typeof result.speed_governor.enabled!=='boolean')throw new Error('地形减速开关无效');
    for(const [key,range] of Object.entries(contract.speed_governor.limits))number(key,...range,false,result.speed_governor);
    if(Object.keys(result.speed_governor).sort().join()!==Object.keys(contract.speed_governor.defaults).sort().join())
      throw new Error('地形减速参数不完整');
  }
  return result;
}
export function preset(name,contract,current){
  const value={...defaults(contract),...current,runtime:{...contract.defaults}};
  const refresh={original:24,responsive:8,sensitive:4,per_frame:1}[name];
  if(refresh===undefined)throw new Error('未知预设');
  value.runtime.refresh_frames=refresh;
  return normalizeSettings(value,contract);
}
export function runtimeEqual(left,right){
  return Boolean(left&&right)&&Object.keys(left).every(key=>left[key]===right[key])&&Object.keys(left).length===Object.keys(right).length;
}
