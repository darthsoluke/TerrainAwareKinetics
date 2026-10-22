import mujoco
import numpy as np
import pytest

from motionbricks.mb_sonic.contact_batch import capture_contacts


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("dim", [1, 3, 4, 6])
def test_contact_columns_match_native_force_bits(cone, dim):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option cone="{cone}"/><worldbody><geom type="plane" size="2 2 .1" condim="{dim}"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" condim="{dim}"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = [.3, -.7, -.2, .4, -.8, .6]
    mujoco.mj_forward(model, data)
    batch = capture_contacts(model, data)
    assert len(batch) == data.ncon and len(batch) > 0
    for index, contact in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        assert force.tobytes() == batch.wrench_contact[index].tobytes()
        np.testing.assert_array_equal(batch.geom_ids[index], contact.geom)
        np.testing.assert_array_equal(batch.frame_contact_from_world[index].ravel(), contact.frame)
        np.testing.assert_array_equal(batch.position_world[index], contact.pos)
    frozen = batch.position_world.copy()
    data.qpos[2] = 5
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(batch.position_world, frozen)
    empty = capture_contacts(model, data)
    assert empty.wrench_contact.shape == (0, 6) and empty.geom_ids.shape == (0, 2)
