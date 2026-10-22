"""Static PFNN-style terrain fitting to offline retargeted support geometry."""

import ast
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def author_functions(folder):
    folder = Path(folder)
    tree = ast.parse((folder/'generate_database.py').read_text(encoding='utf-8-sig'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'patchfunc']
    if len(functions) != 1:
        raise ValueError('Released PFNN patch sampling function required')

    class ModernIntegers(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'int':
                return ast.copy_location(ast.Name(id='int', ctx=ast.Load()), node)
            return self.generic_visit(node)

    module = ast.fix_missing_locations(ModernIntegers().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = {'np': np}
    exec(compile(module, str(folder/'generate_database.py'), 'exec'), namespace)
    specification = importlib.util.spec_from_file_location('_pfnn_author_rbf', folder/'motion/Learning.py')
    implementation = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(implementation)
    return namespace['patchfunc'], implementation.RBF


def terrain_kind(stem):
    if any(name in stem for name in ('LocomotionFlat12_000', 'NewCaptures03_000', 'NewCaptures03_001',
                                    'NewCaptures03_002', 'NewCaptures04_000')):
        return 'jumpy'
    if 'WalkingUpSteps06_000' in stem:
        return 'beam'
    if any(name in stem for name in ('NewCaptures01_000', 'NewCaptures02_000', 'WalkingUpSteps09_000',
                                    'WalkingUpSteps10_000', 'WalkingUpSteps11_000')) or 'Flat' in stem:
        return 'flat'
    return 'rocky'


def footstep_cycles(path, source_fps, frames, fps=30):
    events = []
    for line_number, text in enumerate(Path(path).read_text().splitlines()):
        values = text.split()
        if not values:
            continue
        events.append((int(values[0]), '*' in text, line_number))
    rows = []
    for left, right in zip(events, events[1:]):
        first, last = int(np.floor(left[0]/source_fps*fps)), int(np.floor(right[0]/source_fps*fps))
        if left[1] or right[1] or first-30 < 0 or last+31 > frames or last <= first:
            continue
        rows.append(dict(start=first-30, stop=last+31, core_start=first, core_stop=last,
            original_event_frames=[left[0], right[0]], annotation_lines=[left[2], right[2]]))
    return rows


class FittedTerrain:
    def __init__(self, patchfunc, patch, center, mean, support_mean, correction, unit, metadata):
        self.patchfunc, self.patch, self.center = patchfunc, patch, center
        self.mean, self.support_mean, self.correction = mean, support_mean, correction
        self.unit, self.metadata = unit, metadata

    def __call__(self, world_xz):
        query = np.asarray(world_xz, dtype=np.float64)/self.unit
        if self.patch is None:
            return np.full(len(query), self.support_mean*self.unit)
        basic = self.patchfunc(self.patch[None], query-self.center)[0, :, 0]-self.mean+self.support_mean
        return (basic+self.correction(query)[:, 0])*self.unit


def fit_terrain(patches, support_points, contact_mask, roots, kind, scale, official_folder, seed):
    if support_points.shape[1:] != (4, 3) or contact_mask.shape != support_points.shape[:2]:
        raise ValueError('Four actual retargeted heel/toe support channels required')
    if not np.isfinite(support_points).all() or not np.isfinite(roots).all() or scale <= 0:
        raise ValueError('Finite body geometry and positive retarget scale required')
    down, up = support_points[contact_mask], support_points[~contact_mask]
    if len(down) < 8:
        raise ValueError('Insufficient source-kinematic support observations')
    unit = .01*scale
    down, up = down/unit, up/unit
    center, support_mean = down[:, [0, 2]].mean(0), float(down[:, 1].mean())
    patchfunc, rbf = author_functions(official_folder)
    metadata = dict(kind=kind, support_points=len(down), unit_metres=unit,
        construction='Author patch objective and linear RBF; support anchors are retargeted SOMA skin.',
        measured_scene=False, time_varying_geometry=False)
    if kind == 'flat':
        return FittedTerrain(patchfunc, None, center, 0, support_mean, None, unit, metadata)
    down_height = patchfunc(patches, down[:, [0, 2]]-center)[..., 0]
    down_mean = down_height.mean(1)
    centered_down = down[:, 1]-support_mean
    error = .1*np.square((down_height-down_mean[:, None])-centered_down[None]).mean(1)
    if len(up):
        up_height = patchfunc(patches, up[:, [0, 2]]-center)[..., 0]-down_mean[:, None]
        desired_up = up[:, 1]-support_mean
        error += np.square(np.maximum(up_height-desired_up[None], 0)).mean(1)
        if kind == 'jumpy':
            error += np.square(np.maximum(desired_up[None]-5.-up_height, 0)).mean(1)
    if kind == 'beam':
        rng = np.random.RandomState(seed)
        root_cm = roots/unit
        outside = root_cm+np.array([50., 0., 50.])*rng.normal(size=root_cm.shape)
        far = (np.linalg.norm(outside[:, None]-root_cm[None], axis=-1) > 15.).all(1)
        if far.any():
            center_y = patchfunc(patches, root_cm[:, [0, 2]]-center)[..., 0]
            outside_y = patchfunc(patches, outside[far][:, [0, 2]]-center)[..., 0]
            # Match each sampled outside point to its original time slot.
            error += np.square(np.maximum(outside_y-(center_y[:, far]-40.), 0)).mean(1)
    chosen = int(error.argmin())
    patch = patches[chosen]
    basic = patchfunc(patch[None], down[:, [0, 2]]-center)[0, :, 0]-down_mean[chosen]+support_mean
    correction = rbf(smooth=.1, function='linear')
    correction.fit(down[:, [0, 2]], (down[:, 1]-basic)[:, None])
    metadata.update(patch_index=chosen, objective=float(error[chosen]), rbf_kernel='linear', rbf_smooth=.1)
    return FittedTerrain(patchfunc, patch, center, float(down_mean[chosen]), support_mean, correction, unit, metadata)


def static_heightfield(terrain, roots, spacing=.05, margin=3.):
    if terrain.patch is None:
        low = roots[:, [0, 2]].min(0)-8
        width = float(np.ptp(roots[:, [0, 2]], axis=0).max()+16)
        return dict(heightfield_y_m=np.full((2, 2), terrain.support_mean*terrain.unit, np.float32),
            min_point_world_xyz_m=np.array([low[0], 0., low[1]], np.float32), dx_m=np.float32(width))
    low = np.floor((roots[:, [0, 2]].min(0)-margin)/spacing)*spacing
    high = np.ceil((roots[:, [0, 2]].max(0)+margin)/spacing)*spacing
    counts = np.rint((high-low)/spacing).astype(int)+1
    if int(np.prod(counts)) > 1_000_000:
        raise ValueError('Source episode scene extent exceeds the declared preparation budget')
    x, z = np.meshgrid(low[0]+np.arange(counts[0])*spacing, low[1]+np.arange(counts[1])*spacing)
    query = np.stack((x.ravel(), z.ravel()), -1)
    heights = np.concatenate([terrain(query[start:start+4096]) for start in range(0, len(query), 4096)])
    if not np.isfinite(heights).all():
        raise ValueError('Fitted static terrain is non-finite')
    return dict(heightfield_y_m=heights.reshape(counts[1], counts[0]).astype(np.float32),
        min_point_world_xyz_m=np.array([low[0], 0., low[1]], np.float32), dx_m=np.float32(spacing))
