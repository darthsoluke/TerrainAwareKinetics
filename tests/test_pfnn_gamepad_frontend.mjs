import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import * as THREE from '../web/full_corpus_lab/vendor/three.module.js';

const source = await readFile(new URL('../web/full_corpus_lab/app.js', import.meta.url), 'utf8');
const definition = source.slice(source.indexOf('class KinematicGamepad'), source.indexOf('const gamepadInput'));
const Gamepad = vm.runInNewContext(`${definition}; KinematicGamepad`);
const pad = (x=0,y=0,turn=0,menu=false) => ({id:'Xbox',index:0,connected:true,mapping:'standard',
  axes:[x,y,turn,0],buttons:Array.from({length:10},(_,i)=>({pressed:i===9 && menu}))});

test('connection requires centered sticks; forward, reverse and strafe retain analog magnitude', () => {
  const input = new Gamepad();
  assert.equal(input.sample([pad(0,-1)],true).armed,false);
  assert.equal(input.sample([pad()],true).armed,true);
  assert.equal(input.sample([pad(0,-1)],true).velocity[1],1);
  assert.equal(input.sample([pad(0,1)],true).velocity[1],-1);
  assert.ok(Math.abs(input.sample([pad(.56)],true).velocity[0]-.5)<1e-12);
  assert.ok(Math.abs(Math.hypot(...input.sample([pad(1,-1)],true).velocity)-1)<1e-12);
});

test('deadzone, blur and disconnect stop intent and require recentering', () => {
  const input = new Gamepad();
  assert.equal(input.sample([pad(.04,-.03,.05)],true).armed,true);
  assert.equal(Math.hypot(...input.sample([pad(.04,-.03,.05)],true).velocity),0);
  input.sample([pad(1)],true);
  assert.equal(input.sample([pad(1)],false).velocity[0],0);
  assert.equal(input.sample([pad(1)],true).armed,false);
  input.sample([pad()],true);
  assert.equal(input.sample([pad(1)],true).velocity[0],1);
  assert.equal(input.sample([],true).armed,false);
  assert.equal(input.sample([pad(1)],true).velocity[0],0);
});

test('Menu toggles once per press and does not activate on device discovery', () => {
  const input = new Gamepad();
  assert.equal(input.sample([pad(0,0,0,true)],true).menuPressed,false);
  input.sample([pad()],true);
  assert.equal(input.sample([pad(0,0,0,true)],true).menuPressed,true);
  assert.equal(input.sample([pad(0,0,0,true)],true).menuPressed,false);
});

function commandContext(velocity, turn=0, keys=[]) {
  const state = {keyState:new Set(keys),touchState:new Set(),gamepad:{armed:true,velocity,turn},
    latestCommand:{heading:.7},commandCameraForward:new THREE.Vector3(0,0,1),
    camera:{up:new THREE.Vector3(0,1,0)},tuningExtras:{run_multiplier:1.6},config:{},paused:false,live:{}};
  const context = vm.createContext({THREE,state,isGenerativeMode:()=>true,readSpeed:()=>3,
    maxMovementSpeed:()=>12,setCommand:(v,h)=>{state.latestCommand={velocity_xz:v,heading:h};}});
  vm.runInContext(source.slice(source.indexOf('function updateCommandFromInput('),
    source.indexOf('function normalizeSpeedInput()')),context);
  vm.runInContext('updateCommandFromInput(.1)',context);
  return state.latestCommand;
}

test('left stick strafes/reverses without turning; right stick turns in place', () => {
  for(const velocity of [[1,0],[0,-1],[.5,.5]]) {
    const command=commandContext(velocity);
    assert.equal(command.heading,.7);
    assert.ok(Math.abs(Math.hypot(...command.velocity_xz)-3*Math.hypot(...velocity))<1e-12);
  }
  const turn=commandContext([0,0],1);
  assert.ok(Math.abs(turn.heading-.88)<1e-12);
  assert.equal(Math.hypot(...turn.velocity_xz),0);
  const combined=commandContext([0,-1],-1);
  assert.ok(Math.abs(combined.heading-.52)<1e-12);
  assert.equal(combined.velocity_xz[1],-3);
});

test('keyboard movement takes priority and preserves existing camera-relative heading', () => {
  const command=commandContext([1,0],1,['w']);
  assert.equal(command.heading,0);
  assert.equal(command.velocity_xz[1],3);
});
