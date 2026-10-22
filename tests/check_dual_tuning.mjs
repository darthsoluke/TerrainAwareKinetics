import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const source=await fs.readFile(new URL('../web/full_corpus_lab/tuning-contract.js',import.meta.url),'utf8');
const {normalizeSettings,SCHEMA}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const old=JSON.parse(await fs.readFile(new URL('../out/pfnn_mixed_20261015/gait_preproof_service_v1.json',import.meta.url),'utf8')).current_server_tuning.runtime;
const contract=structuredClone(old);
contract.planning_policy={defaults:{schema:'motionbricks.dual_horizon_execution.v1',lookahead_tokens:8,execute_tokens:8},lookahead_choices:[8,12,16]};
contract.limits.num_tokens=[8,8];contract.limits.inference_steps=[4,4];contract.limits.refresh_frames=[4];
for(const tokens of [8,12,16]){
  const saved={schema:SCHEMA,runtime:{...old.defaults,num_tokens:tokens},speed_mps:6,run_multiplier:1.6,
    mouse_sensitivity:1.3,field_of_view:58,invert_y:true,
    speed_governor:{...old.speed_governor.defaults,enabled:true,strength:.7}};
  const result=normalizeSettings(saved,contract);
  assert.equal(result.planning_policy.lookahead_tokens,tokens);
  assert.equal(result.runtime.num_tokens,8);assert.equal(result.runtime.inference_steps,4);assert.equal(result.runtime.refresh_frames,4);
  for(const key of ['speed_mps','run_multiplier','mouse_sensitivity','field_of_view','invert_y','speed_governor'])assert.deepEqual(result[key],saved[key]);
  assert.equal(saved.runtime.num_tokens,tokens);
}
console.log('3 old-profile migrations preserve speed/Shift/governor/camera; execute8 and lookaheadN are distinct.');
