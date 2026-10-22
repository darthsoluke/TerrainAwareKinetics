import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../web/full_corpus_lab/tuning-contract.js',import.meta.url),'utf8');
const {SCHEMA,defaults,normalizeSettings,preset,runtimeEqual}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const contract={max_speed_mps:12,defaults:{num_tokens:8,inference_steps:4,lookahead_m:-1,
  lateral_halfwidth_m:.6,refresh_frames:24,lookahead_gain:1,lookahead_base_m:.4,path_refinements:0},
  limits:{num_tokens:[6,16],inference_steps:[1,8],refresh_frames:[1,2,4,8,16,24],lookahead_m:[.8,20],
    lateral_halfwidth_m:[.3,6],lookahead_gain:[0,4],lookahead_base_m:[0,10]}};
test('presets preserve camera and speed but choose explicit refresh cadence',()=>{
  const original={...defaults(contract),speed_mps:3,field_of_view:62};
  for(const [name,refresh] of Object.entries({original:24,responsive:8,sensitive:4,per_frame:1})){
    const value=preset(name,contract,original);
    assert.equal(value.runtime.refresh_frames,refresh);assert.equal(value.speed_mps,3);assert.equal(value.field_of_view,62);
  }
  assert.equal(original.runtime.refresh_frames,24);
});
test('JSON roundtrip retains every supported control',()=>{
  const value=preset('sensitive',contract,defaults(contract));
  assert.deepEqual(normalizeSettings(JSON.parse(JSON.stringify(value)),contract),value);
  assert.equal(value.schema,SCHEMA);
});
test('invalid imports cannot enable refinement or inconsistent generation windows',()=>{
  for(const patch of [{refresh_frames:3},{num_tokens:6,refresh_frames:24},{path_refinements:1},
    {lookahead_m:NaN},{inference_steps:9},{num_tokens:8.5},{unexpected:1}]){
    const value=defaults(contract);Object.assign(value.runtime,patch);
    assert.throws(()=>normalizeSettings(value,contract));
  }
});
test('camera and movement inputs remain finite and bounded',()=>{
  for(const patch of [{speed_mps:13},{run_multiplier:0},{field_of_view:91},
    {mouse_sensitivity:NaN},{invert_y:1},{schema:'other'}])
    assert.throws(()=>normalizeSettings({...defaults(contract),...patch},contract));
});
test('requested settings are distinct from actual model settings',()=>{
  assert.equal(runtimeEqual(contract.defaults,{...contract.defaults}),true);
  assert.equal(runtimeEqual(contract.defaults,{...contract.defaults,refresh_frames:4}),false);
  assert.equal(runtimeEqual(contract.defaults,null),false);
});
