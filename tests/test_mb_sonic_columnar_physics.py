"""Columnar fidelity checks; no model construction, simulation or inference."""

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import ClassVar

import numpy as np
import pytest

from motionbricks.mb_sonic.columnar_physics import (
    CONTACT_ARRAY_SHAPES, CONTACT_BATCH_KIND, CONTACT_COLUMN_SHAPES, DENSE_FIELDS, SCHEMA, WARNING_NAMES,
    pack_physics_states, unpack_physics_states,
)
from motionbricks.mb_sonic.runtime import (
    ContactForce, NumericArtifactWriter, PhysicsRecorder, PhysicsState,
    RecordingBudget, _numeric_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_RECORDING = ROOT / "out/mb_sonic_e0_e3_20261018/tracking_isolation_v1/protomotions_matching_robot_v2/recording.jsonl"
REPORT = ROOT / "out/tracking_isolation_v1/columnar_physics_v1.json"
REPORT_V2 = ROOT / "out/tracking_isolation_v1/columnar_physics_v2.json"


@dataclass(frozen=True)
class ContactBatchFixture:
    """Same six-column contract as the parent's pending ContactBatch class."""

    kind: ClassVar[str] = CONTACT_BATCH_KIND
    geom_ids: np.ndarray
    body_ids: np.ndarray
    position_world: np.ndarray
    frame_contact_from_world: np.ndarray
    wrench_contact: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.distance)

    def __iter__(self):
        raise AssertionError("Columnar packing must not iterate ContactBatch")

    def __getitem__(self, _):
        raise AssertionError("Columnar packing must not materialize individual contacts")


def _batch_states(states, batch_type=ContactBatchFixture):
    result = []
    dtypes = {name: value.dtype for name, value in pack_physics_states(states)["contacts"].items()}
    for state in states:
        columns = pack_physics_states((state,))["contacts"]
        result.append(replace(state, contacts=batch_type(**{
            name: columns[name].astype(dtypes[name], copy=False) for name in CONTACT_COLUMN_SHAPES
        })))
    return tuple(result)


def _assert_equal(expected, actual):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype.str == expected.dtype.str
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, float):
        assert np.float64(actual).tobytes() == np.float64(expected).tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for original, restored in zip(expected, actual):
            _assert_equal(original, restored)
    elif isinstance(expected, (PhysicsState, ContactForce)):
        for field in fields(expected):
            _assert_equal(getattr(expected, field.name), getattr(actual, field.name))
    else:
        assert actual == expected


def _fixture(dtype):
    if np.dtype(dtype).itemsize == 8:
        bits = np.array([0, 0x8000000000000000, 0x7ff8000000001234,
                         0xfff8000000005678, 0x7ff0000000000000, 0xfff0000000000000], dtype=np.uint64)
        values = bits.view(np.float64).astype(dtype)
    else:
        bits = np.array([0, 0x80000000, 0x7fc01234, 0xffc05678,
                         0x7f800000, 0xff800000], dtype=np.uint32)
        values = bits.view(np.float32).astype(dtype)

    def array(shape):
        return np.resize(values, shape).astype(dtype)

    contact = ContactForce((2, 9), (1, 7), array((3,)), array((3, 3)), array((6,)),
                           float(np.array([0x7ff800000000babe], dtype=np.uint64).view(np.float64)[0]))
    shapes = dict(qpos=(36,), qvel=(35,), joint_positions=(29,), joint_velocities=(29,),
                  body_positions_world=(34, 3), body_quat_wxyz=(34, 4), actuator_controls=(29,),
                  actuator_forces=(29,), generalized_actuator_forces=(35,), root_qfrc_actuator=(6,),
                  qfrc_applied=(35,), xfrc_applied=(34, 6))
    base = PhysicsState(
        **{name: array(shape) for name, shape in shapes.items()},
        time_s=-0., captured_at_monotonic_ns=2**60 + 13, state_version=2**54 + 7,
        solver_warnings={name: dict(number=index, lastinfo=2**55 + index) for index, name in enumerate(WARNING_NAMES)},
        root_ground_known=True, root_ground_height_m=contact.distance,
        contacts=(contact, replace(contact, geom_ids=(8, 3), body_ids=(4, 2), distance=-0.)),
        terminal_reason=None,
    )
    return (base, replace(base, time_s=float("inf"), contacts=(), root_ground_known=False,
                          root_ground_height_m=None, terminal_reason="nonfinite_actual_state"),
            replace(base, time_s=float("-inf"), contacts=(contact,), root_ground_height_m=-0.))


def _decode_artifact(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"array", "shape", "dtype"}:
                    array = archive[value["array"]].copy()
                    assert list(array.shape) == value["shape"]
                    assert array.dtype.str == value["dtype"]
                    return array
                return {key: decode(child) for key, child in value.items()}
            if isinstance(value, list):
                return [decode(child) for child in value]
            return value

        return decode(metadata["record"]), metadata


def _persist(tmp_path, record):
    budget = RecordingBudget(16 * 1024**2, 4 * 1024**2, 4 * 1024**2)
    with PhysicsRecorder(tmp_path / "receipt.jsonl", {"test": "columnar_physics"}) as recorder:
        writer = NumericArtifactWriter(tmp_path / "numeric", recorder, pending_plans=1,
                                       storage_budget=budget, max_record_bytes=4 * 1024**2)
        writer._persist(("columnar_test", _numeric_snapshot(record), 0))
        writer.close()
    return next((tmp_path / "numeric").glob("*.npz"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f8"])
def test_ragged_nonfinite_and_npz_are_bitwise_lossless(dtype, tmp_path):
    states = _fixture(dtype)
    batch = pack_physics_states(states)
    assert batch["schema"] == SCHEMA
    assert batch["qpos"].dtype.str == np.dtype(dtype).str
    assert batch["solver_warning_values"].shape == (3, 7, 2)
    np.testing.assert_array_equal(batch["contacts"]["offsets"], [0, 2, 2, 3])
    assert batch["root_ground_height_none"].tolist() == [False, True, False]
    assert batch["terminal_reason"] == [None, "nonfinite_actual_state", None]
    for name in ("time_s", "root_ground_height_m"):
        assert batch[name].dtype == np.float64
    for name in ("captured_at_monotonic_ns", "state_version", "solver_warning_values"):
        assert batch[name].dtype == np.int64
    assert batch["root_ground_known"].dtype == np.bool_
    _assert_equal(states, unpack_physics_states(batch))
    restored, _ = _decode_artifact(_persist(tmp_path, batch))
    _assert_equal(states, unpack_physics_states(restored))
    for name in DENSE_FIELDS:
        assert not np.shares_memory(batch[name], getattr(states[0], name))
        assert not np.shares_memory(getattr(unpack_physics_states(batch)[0], name), batch[name])


def test_all_empty_contacts_and_empty_state_batch():
    states = tuple(replace(state, contacts=()) for state in _fixture(np.float64))
    batch = pack_physics_states(states)
    columns = batch["contacts"]
    np.testing.assert_array_equal(columns["offsets"], [0, 0, 0, 0])
    for name, shape in dict(CONTACT_ARRAY_SHAPES, geom_ids=(2,), body_ids=(2,), distance=()).items():
        assert columns[name].shape == (0, *shape)
    _assert_equal(states, unpack_physics_states(batch))
    assert unpack_physics_states(pack_physics_states(())) == ()


def test_mixed_column_dtype_or_shape_is_rejected_without_promotion():
    states = _fixture(np.float64)
    with pytest.raises(ValueError, match="no promotion"):
        pack_physics_states((states[0], replace(states[1], qpos=states[1].qpos.astype(np.float32))))
    with pytest.raises(ValueError, match="no promotion"):
        pack_physics_states((states[0], replace(states[1], qpos=states[1].qpos[:-1])))
    with pytest.raises(ValueError, match="seven"):
        pack_physics_states((replace(states[0], solver_warnings={}),))
    with pytest.raises(ValueError, match="schema"):
        unpack_physics_states(dict(schema="unknown"))


def _load_real_tick():
    with REAL_RECORDING.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("kind") == "policy_tick" and len(record["physics_states"]) == 20:
                states = []
                for row in record["physics_states"]:
                    contacts = tuple(ContactForce(
                        geom_ids=tuple(contact["geom_ids"]), body_ids=tuple(contact["body_ids"]),
                        **{name: np.asarray(contact[name], dtype=np.float64) for name in CONTACT_ARRAY_SHAPES},
                        distance=contact["distance"],
                    ) for contact in row["contacts"])
                    states.append(PhysicsState(**dict(
                        row, **{name: np.asarray(row[name], dtype=np.float64) for name in DENSE_FIELDS},
                        contacts=contacts)))
                return record, tuple(states), dict(path=str(REAL_RECORDING), line=number,
                    record_sha256=hashlib.sha256(line).hexdigest(), record_bytes=len(line))
    raise AssertionError("No existing 20-substep policy tick found")


def _array_count(value):
    if isinstance(value, np.ndarray):
        return 1
    if isinstance(value, dict):
        return sum(_array_count(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_count(child) for child in value)
    return 0


def _timing(values):
    array = np.asarray(values, dtype=np.float64) / 1e6
    return {"p50": float(np.percentile(array, 50)), "p99": float(np.percentile(array, 99)),
            "max": float(array.max())}


def test_existing_real_twenty_substep_tick_and_cpu_microprofile(tmp_path):
    record, states, source = _load_real_tick()
    batch = pack_physics_states(states)
    _assert_equal(states, unpack_physics_states(batch))
    packed_record = dict(record, physics_states=batch)
    artifact = _persist(tmp_path, packed_record)
    restored, metadata = _decode_artifact(artifact)
    _assert_equal(states, unpack_physics_states(restored["physics_states"]))
    assert artifact.stat().st_size < 4 * 1024**2
    pack_ns, unpack_ns = [], []
    for _ in range(100):
        started = perf_counter_ns()
        packed = pack_physics_states(states)
        pack_ns.append(perf_counter_ns() - started)
        started = perf_counter_ns()
        unpacked = unpack_physics_states(packed)
        unpack_ns.append(perf_counter_ns() - started)
    _assert_equal(states, unpacked)
    module = ROOT / "motionbricks/mb_sonic/columnar_physics.py"
    report = dict(schema="motionbricks.columnar_physics_audit.v1", status="PASS", source=source,
        states=len(states), contacts=sum(len(state.contacts) for state in states),
        array_count=_array_count(batch), full_tick_npz_bytes=artifact.stat().st_size,
        full_tick_npz_array_count=_array_count(restored), max_record_bytes=4 * 1024**2,
        metadata_json_bytes=len(json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()),
        bitwise_fields=[field.name for field in fields(PhysicsState)],
        bitwise_contact_fields=[field.name for field in fields(ContactForce)],
        npz_reader_roundtrip="PASS", measured_iterations=100, pack_ms=_timing(pack_ns),
        unpack_ms=_timing(unpack_ns), simulation_steps=0, gpu_seconds=0, optimizer_steps=0,
        source_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f8"])
def test_contact_batch_nonfinite_bitwise_without_iteration(dtype):
    states = _fixture(dtype)
    batched = _batch_states(states)
    expected = pack_physics_states(states)
    packed = pack_physics_states(batched)
    _assert_equal(expected, packed)
    _assert_equal(states, unpack_physics_states(packed))
    for name in CONTACT_COLUMN_SHAPES:
        assert not np.shares_memory(packed["contacts"][name], getattr(batched[0].contacts, name))


def test_contact_batch_empty_mixed_legacy_and_int32_ids():
    states = _fixture(np.float64)
    batched = _batch_states(states)
    mixed = (batched[0], states[1], states[2])
    _assert_equal(pack_physics_states(states), pack_physics_states(mixed))
    empty = tuple(replace(state, contacts=()) for state in states)
    _assert_equal(pack_physics_states(empty), pack_physics_states(_batch_states(empty)))
    ids32 = tuple(replace(state, contacts=replace(state.contacts,
        geom_ids=state.contacts.geom_ids.astype(np.int32),
        body_ids=state.contacts.body_ids.astype(np.int32))) for state in batched)
    packed = pack_physics_states(ids32)
    assert packed["contacts"]["geom_ids"].dtype == np.int32
    assert packed["contacts"]["body_ids"].dtype == np.int32
    _assert_equal(states, unpack_physics_states(packed))


def test_contact_batch_invalid_shape_and_dtype_are_not_promoted():
    states = _batch_states(_fixture(np.float64))
    invalid = replace(states[0].contacts, position_world=np.empty((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        pack_physics_states((replace(states[0], contacts=invalid),))
    invalid = replace(states[2].contacts, wrench_contact=states[2].contacts.wrench_contact.astype(np.float32))
    with pytest.raises(ValueError, match="no promotion"):
        pack_physics_states((states[0], replace(states[2], contacts=invalid)))


def test_contact_batch_real_tick_pack_microprofile():
    _, states, source = _load_real_tick()
    batched = _batch_states(states)
    expected = pack_physics_states(states)
    _assert_equal(expected, pack_physics_states(batched))
    pack_ns = []
    for _ in range(100):
        started = perf_counter_ns()
        packed = pack_physics_states(batched)
        pack_ns.append(perf_counter_ns() - started)
    _assert_equal(expected, packed)
    module = ROOT / "motionbricks/mb_sonic/columnar_physics.py"
    baseline = json.loads(REPORT.read_text(encoding="utf-8"))
    measured = _timing(pack_ns)
    report = dict(schema="motionbricks.columnar_physics_audit.v2", status="PASS", source=source,
        contact_batch_kind=CONTACT_BATCH_KIND, tested_class="ContactBatchFixture",
        batch_iteration_allowed=False, batch_indexing_allowed=False,
        states=len(states), contacts=sum(len(state.contacts) for state in states),
        array_count=_array_count(packed), measured_iterations=100, pack_ms=measured,
        bitwise_against_legacy="PASS", output_schema=SCHEMA, unpack_changed=False,
        baseline_report=str(REPORT), baseline_pack_ms=baseline["pack_ms"],
        p50_speedup_vs_v1=baseline["pack_ms"]["p50"] / measured["p50"],
        simulation_steps=0, gpu_seconds=0, optimizer_steps=0,
        source_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    REPORT_V2.parent.mkdir(parents=True, exist_ok=True)
    REPORT_V2.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
