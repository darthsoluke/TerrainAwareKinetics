"""Author the same bounded flat test domain used by the controller comparison."""

import json
from pathlib import Path

import numpy as np

from motionbricks.data.bones_contract import file_sha256


def main():
    folder = Path(__file__).resolve().parents[1]/'out/a100_live_20261010'
    original = folder/'flat.npz'
    source_hash = '6823d99cd4a68e35874242978d898c6ba68a577bb473538f5acb3855d2344406'
    if file_sha256(original) != source_hash:
        raise ValueError('Original seed artifact changed')
    output = folder/'flat_recovery_20261013/flat100m.npz'
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Preserve the existing test-scene artifact')
    with np.load(original, allow_pickle=False) as source:
        if not np.all(source['heightfield_y_m']==0):
            raise ValueError('Expected the original zero-plane seed')
        history = source['initial_history'].copy()
        metadata = json.loads(str(source['metadata_json']))
    metadata.update(authored_test_geometry=True, training_data=False,
        geometry_origin='New authored100m flat plane,not inferred missing real terrain',
        source_scene_sha256=source_hash, source_history_modified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, initial_history=history, heightfield_y_m=np.zeros((201,201), np.float32),
        min_point_world_xyz_m=np.array([-50.,0.,-50.], np.float32), dx_m=np.array(.5, np.float32),
        metadata_json=np.array(json.dumps(metadata)))
    with np.load(output, allow_pickle=False) as check:
        np.testing.assert_array_equal(check['initial_history'], history)
    receipt = dict(metadata, scene_sha256=file_sha256(output))
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
