"""Export a browser visual-retarget preview for a skinned human GLB.

The exported viewer uses Three.js to load a rigged humanoid GLB, then applies a
conservative visual retarget from official MotionBricks G1 qpos rollouts onto
the GLB bones at runtime. This is a preview bridge for inspecting motion on a
human-shaped rig; it is not a physical human controller or validated embodiment.
"""

import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch as t

from motionbricks.motion_backbone.demo.utils import navigation_demo
from render_human_skeleton_demo import HUMAN_JOINTS, _retarget_frame


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MotionBricks Visual Retarget Preview</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #101217; }
    #hud {
      position: fixed; left: 16px; top: 14px; z-index: 2; color: #f1f5f9;
      font: 13px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: rgba(16,18,23,.72); padding: 10px 12px; border: 1px solid rgba(255,255,255,.12);
      border-radius: 8px; backdrop-filter: blur(10px);
      max-width: 360px;
    }
    #hud b { color: #93c5fd; }
    #hud .muted { color: #cbd5e1; }
    #hud .notice { color: #fbbf24; margin-top: 3px; }
    #hud label { display: block; margin-top: 8px; cursor: pointer; user-select: none; }
    #hud input { margin-right: 6px; vertical-align: -1px; }
  </style>
</head>
<body>
  <div id="hud">
    <div><b>MotionBricks visual retarget preview</b></div>
    <div class="muted">official G1 qpos -> human rig / skinned GLB</div>
    <div class="notice">Visual preview only; not physical human control.</div>
    <label><input id="debugToggle" type="checkbox" />debug skeleton / bone axes</label>
    <div class="muted">Space: pause/play - R: reset camera</div>
    <div id="status" class="muted"></div>
  </div>
  <script type="importmap">
    {
      "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/"
      }
    }
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const debugToggle = document.getElementById('debugToggle');
    const statusEl = document.getElementById('status');

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101217);
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
    camera.position.set(2.8, 1.7, 5.2);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.0, 0);
    controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x253040, 2.2);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 3.0);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    scene.add(key);

    const grid = new THREE.GridHelper(12, 48, 0x496477, 0x23313f);
    scene.add(grid);

    const [motion, gltf] = await Promise.all([
      fetch('./animation.json').then(r => r.json()),
      new GLTFLoader().loadAsync('./avatar.glb'),
    ]);

    const avatar = gltf.scene;
    avatar.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
        if (obj.material) {
          obj.material.roughness = 0.72;
          obj.material.metalness = 0.05;
        }
      }
    });
    scene.add(avatar);
    avatar.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(avatar);
    const avatarCenter = bounds.getCenter(new THREE.Vector3());
    const avatarSize = bounds.getSize(new THREE.Vector3()).length();
    controls.target.copy(avatarCenter);
    controls.target.y += 0.2;
    camera.position.copy(avatarCenter).add(new THREE.Vector3(0, avatarSize * 0.28, avatarSize * 2.05));
    const cameraOffset = camera.position.clone().sub(controls.target);
    camera.near = Math.max(0.01, avatarSize / 200);
    camera.far = Math.max(100, avatarSize * 20);
    camera.updateProjectionMatrix();

    const bones = {};
    const boneList = [];
    const boneByName = new Map();
    const boneByKey = new Map();
    const objectsByName = new Map();
    const objectsByKey = new Map();

    function boneKey(name) {
      return String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
    }

    avatar.traverse((obj) => {
      if (obj.name) {
        objectsByName.set(obj.name, obj);
        const key = boneKey(obj.name);
        if (!objectsByKey.has(key)) objectsByKey.set(key, obj);
      }
      if (obj.isBone) {
        bones[obj.name] = obj;
        boneList.push(obj);
        boneByName.set(obj.name, obj);
        const key = boneKey(obj.name);
        if (!boneByKey.has(key)) boneByKey.set(key, obj);
      }
    });

    function resolveBone(ref) {
      if (!ref) return null;
      const names = Array.isArray(ref) ? ref : [ref];
      for (const name of names) {
        if (boneByName.has(name)) return boneByName.get(name);
      }
      for (const name of names) {
        const key = boneKey(name);
        if (boneByKey.has(key)) return boneByKey.get(key);
      }
      return null;
    }

    function resolveObject(ref) {
      if (!ref) return null;
      const names = Array.isArray(ref) ? ref : [ref];
      for (const name of names) {
        if (objectsByName.has(name)) return objectsByName.get(name);
      }
      for (const name of names) {
        const key = boneKey(name);
        if (objectsByKey.has(key)) return objectsByKey.get(key);
      }
      return null;
    }

    function refLabel(ref) {
      return Array.isArray(ref) ? ref[0] : ref;
    }

    const tmpA = new THREE.Vector3();
    const tmpB = new THREE.Vector3();
    const tmpQ = new THREE.Quaternion();
    const tmpParentQ = new THREE.Quaternion();
    const retargetLinks = [];
    const missingLinks = [];

    for (const link of motion.boneMap) {
      const [boneRef, targetStart, targetEnd, childRef] = link;
      const bone = resolveBone(boneRef);
      const child = resolveObject(childRef);
      if (!bone || !child) {
        missingLinks.push(refLabel(boneRef));
        continue;
      }
      retargetLinks.push({
        label: refLabel(boneRef),
        bone,
        child,
        targetStart,
        targetEnd,
        bindDir: null,
        bindWorldQuat: null,
      });
    }

    avatar.updateMatrixWorld(true);
    for (const link of retargetLinks) {
      const { bone, child } = link;
      bone.getWorldQuaternion(tmpQ);
      link.bindWorldQuat = tmpQ.clone();
      bone.getWorldPosition(tmpA);
      child.getWorldPosition(tmpB);
      if (tmpB.distanceTo(tmpA) < 1e-5) {
        link.bindDir = null;
        continue;
      }
      link.bindDir = tmpB.sub(tmpA).normalize().clone();
    }

    const skeletonHelper = new THREE.SkeletonHelper(avatar);
    skeletonHelper.visible = false;
    skeletonHelper.material.depthTest = false;
    scene.add(skeletonHelper);

    const axesHelpers = [];
    for (const link of retargetLinks) {
      const axes = new THREE.AxesHelper(0.08);
      axes.name = `RetargetAxes.${link.label}`;
      axes.visible = false;
      link.bone.add(axes);
      axesHelpers.push(axes);
    }

    function setDebugVisible(visible) {
      skeletonHelper.visible = visible;
      for (const axes of axesHelpers) axes.visible = visible;
    }
    debugToggle.addEventListener('change', () => setDebugVisible(debugToggle.checked));

    statusEl.textContent = `Mapped ${retargetLinks.filter(link => link.bindDir).length}/${motion.boneMap.length} retarget links`;
    if (missingLinks.length) {
      console.warn('Missing retarget bones:', missingLinks);
    }

    window.motionViewer = { scene, avatar, camera, controls, bones, boneList, motion, retargetLinks };

    function humanVec(frame, name) {
      const v = frame[name];
      return new THREE.Vector3(v[0], v[1], v[2]);
    }

    function setBoneWorldQuaternion(bone, qWorld) {
      if (bone.parent) {
        bone.parent.getWorldQuaternion(tmpParentQ);
        bone.quaternion.copy(tmpParentQ.invert().multiply(qWorld));
      } else {
        bone.quaternion.copy(qWorld);
      }
      bone.updateMatrixWorld(true);
    }

    const basePelvis = humanVec(motion.frames[0], 'pelvis');
    const rootBone = resolveBone(['Bone', 'Root', 'Armature', 'Hips', 'mixamorigHips']) || avatar;
    const rootStart = rootBone.position.clone();
    const rootFollow = new THREE.Vector3();
    const scale = 1.0;

    function applyFrame(frame) {
      if (rootBone) {
        const pelvis = humanVec(frame, 'pelvis').sub(basePelvis).multiplyScalar(scale);
        rootBone.position.copy(rootStart).add(new THREE.Vector3(0, pelvis.y, 0));
      }
      avatar.updateMatrixWorld(true);
      rootFollow.set(0, 0, 0);

      for (const link of retargetLinks) {
        const { bone, targetStart, targetEnd, bindDir, bindWorldQuat } = link;
        if (!bindDir || !bindWorldQuat) continue;
        const desired = humanVec(frame, targetEnd).sub(humanVec(frame, targetStart)).normalize();
        if (desired.lengthSq() < 1e-8) continue;
        const delta = new THREE.Quaternion().setFromUnitVectors(bindDir, desired);
        const qWorld = delta.multiply(bindWorldQuat.clone());
        setBoneWorldQuaternion(bone, qWorld);
      }
    }

    let paused = false;
    let frameIdx = 0;
    let last = performance.now();
    addEventListener('keydown', (event) => {
      if (event.code === 'Space') paused = !paused;
      if (event.code === 'KeyR') {
        controls.target.copy(avatarCenter);
        controls.target.y += 0.2;
        camera.position.copy(avatarCenter).add(new THREE.Vector3(0, avatarSize * 0.28, avatarSize * 2.05));
      }
    });

    function animate(now) {
      requestAnimationFrame(animate);
      const frameDuration = 1000 / motion.fps;
      if (!paused && now - last > frameDuration) {
        frameIdx = (frameIdx + 1) % motion.frames.length;
        applyFrame(motion.frames[frameIdx]);
        last = now;
      }
      const followTarget = avatarCenter.clone().add(rootFollow);
      followTarget.y += 0.2;
      controls.target.lerp(followTarget, 0.16);
      camera.position.lerp(followTarget.clone().add(cameraOffset), 0.16);
      controls.update();
      renderer.render(scene, camera);
    }

    applyFrame(motion.frames[0]);
    requestAnimationFrame(animate);

    addEventListener('resize', () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    });
  </script>
</body>
</html>
"""


BONE_MAP = [
    [["Hips", "mixamorigHips"], "pelvis", "spine", ["Abdomen", "Spine", "mixamorigSpine"]],
    [["Abdomen", "Spine", "mixamorigSpine"], "spine", "chest", ["Torso", "Chest", "mixamorigSpine1"]],
    [["Torso", "Chest", "mixamorigSpine1"], "chest", "neck", ["Neck", "mixamorigNeck"]],
    [["Neck", "mixamorigNeck"], "neck", "head", ["Head", "mixamorigHead"]],
    [
        ["UpperArm.L", "UpperArmL", "LeftArm", "mixamorigLeftArm"],
        "left_shoulder",
        "left_elbow",
        ["LowerArm.L", "LowerArmL", "LeftForeArm", "mixamorigLeftForeArm"],
    ],
    [
        ["LowerArm.L", "LowerArmL", "LeftForeArm", "mixamorigLeftForeArm"],
        "left_elbow",
        "left_wrist",
        ["Palm.L", "Hand.L", "LeftHand", "mixamorigLeftHand"],
    ],
    [
        ["UpperArm.R", "UpperArmR", "RightArm", "mixamorigRightArm"],
        "right_shoulder",
        "right_elbow",
        ["LowerArm.R", "LowerArmR", "RightForeArm", "mixamorigRightForeArm"],
    ],
    [
        ["LowerArm.R", "LowerArmR", "RightForeArm", "mixamorigRightForeArm"],
        "right_elbow",
        "right_wrist",
        ["Palm.R", "Hand.R", "RightHand", "mixamorigRightHand"],
    ],
    [
        ["UpperLeg.L", "UpperLegL", "LeftUpLeg", "mixamorigLeftUpLeg"],
        "left_hip",
        "left_knee",
        ["LowerLeg.L", "LowerLegL", "LeftLeg", "mixamorigLeftLeg"],
    ],
    [
        ["LowerLeg.L", "LowerLegL", "LeftLeg", "mixamorigLeftLeg"],
        "left_knee",
        "left_ankle",
        ["LowerLeg.L_end", "LowerLegLEnd", "Foot.L", "FootL", "LeftFoot", "mixamorigLeftFoot"],
    ],
    [
        ["UpperLeg.R", "UpperLegR", "RightUpLeg", "mixamorigRightUpLeg"],
        "right_hip",
        "right_knee",
        ["LowerLeg.R", "LowerLegR", "RightLeg", "mixamorigRightLeg"],
    ],
    [
        ["LowerLeg.R", "LowerLegR", "RightLeg", "mixamorigRightLeg"],
        "right_knee",
        "right_ankle",
        ["LowerLeg.R_end", "LowerLegREnd", "Foot.R", "FootR", "RightFoot", "mixamorigRightFoot"],
    ],
]


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_agent = navigation_demo(args)
    demo_agent.full_agent.reset()
    names = demo_agent.full_agent._motion_rep.skeleton.bone_order_names
    frames = []

    for step in range(args.max_steps):
        force_idle = step + 30 > args.max_steps
        qpos = demo_agent.full_agent.get_next_frame()
        context_motion_features = demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        control_signals = demo_agent.controller.generate_control_signals(
            None,
            demo_agent.mj_model,
            demo_agent.mj_data,
            visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": args.allowed_mode},
        )
        if args.use_qpos:
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        else:
            control_signals["context_motion_features"] = context_motion_features

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
            )

        if step % args.frame_stride != 0:
            continue

        qpos_tensor = t.as_tensor(qpos, dtype=t.float32, device=demo_agent.full_agent._device).view(1, 1, -1)
        g1_positions, _ = demo_agent.full_agent._converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
        human = _retarget_frame(g1_positions[0, 0].detach().cpu().numpy(), names)
        frames.append({name: [float(x) for x in human[name]] for name in HUMAN_JOINTS})

    animation = {
        "fps": float(demo_agent.inferencer.motion_rep.fps / args.frame_stride),
        "frames": frames,
        "boneMap": BONE_MAP,
        "source": "official MotionBricks G1 qpos rollout visual-retargeted to a humanoid GLB rig",
        "disclaimer": "Visual retarget preview only; not physical human control or validated human embodiment.",
    }
    (out_dir / "animation.json").write_text(json.dumps(animation), encoding="utf-8")
    shutil.copyfile(args.avatar_glb, out_dir / "avatar.glb")
    (out_dir / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote skinned viewer to {out_dir / 'index.html'}")
    print(f"Wrote {len(frames)} retargeted frames to {out_dir / 'animation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Three.js visual-retarget preview for MotionBricks")
    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    parser.add_argument("--controller", type=str, default="random", choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="out/skinned_human_viewer")
    parser.add_argument(
        "--avatar_glb",
        type=str,
        default="assets/human_cc0/animated_men_pack_glb/Man.glb",
        help="Rigged humanoid GLB. The default is the downloaded CC0 Quaternius character.",
    )
    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    start = time.time()
    main(args)
    print(f"Done in {time.time() - start:.1f}s")
