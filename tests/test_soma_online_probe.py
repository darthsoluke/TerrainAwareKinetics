import asyncio
import json

from aiohttp import web
import numpy as np
import pytest

from scripts import probe_soma_online_rollout as probe


def test_phase_labels_and_straight_command():
    assert [probe.phase_for_block(index) for index in range(8)] == [
        "idle", "idle", "idle", "idle", "idle", "idle", "straight", "straight"]
    assert probe.command_for_phase("idle", 1.25) == {
        "velocity_xz": [0.0, 0.0], "heading": 0.0}
    assert probe.command_for_phase("straight", 1.25) == {
        "velocity_xz": [0.0, 1.25], "heading": 0.0}
    with pytest.raises(ValueError, match="Unknown phase"):
        probe.command_for_phase("walk", 1.0)


def test_decode_published_block_checks_frames_stride_and_complete_binary():
    metadata = {"frame_count": 8, "vertex_count": 2, "frame_stride_floats": 9,
                "encoding": "float32-le"}
    source = np.arange(8 * 9, dtype="<f4")
    decoded = probe.decode_published_block(metadata, source.tobytes(), 2)
    assert decoded.shape == (8, 9)
    np.testing.assert_array_equal(decoded, source.reshape(8, 9))

    for changed, payload in (
            ({**metadata, "frame_count": 7}, source.tobytes()),
            ({**metadata, "frame_stride_floats": 8}, source.tobytes()),
            (metadata, source[:-1].tobytes()),
            (metadata, source.tobytes() + b"\x00\x00\x00\x00")):
        with pytest.raises(ValueError):
            probe.decode_published_block(changed, payload, 2)

    nonfinite = source.copy()
    nonfinite[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        probe.decode_published_block(metadata, nonfinite.tobytes(), 2)


def test_debug_trace_uses_matching_published_eight_and_missing_is_not_filled():
    trace = [[float(index), float(index + 10), float(index + 20)] for index in range(12)]
    result = probe.debug_xyz_trace({"root_stage_world_xyz_m": trace,
                                    "published_frame_count": 8},
                                   "root_stage_world_xyz_m")
    assert result["status"] == "ok" and result["available_frames"] == 12
    assert len(result["values"]) == 8 and result["values"][-1] == [7.0, 17.0, 27.0]
    missing = probe.debug_xyz_trace({}, "root_stage_world_xyz_m")
    assert missing == {"status": "missing", "available_frames": None, "values": None}
    mismatch = probe.debug_xyz_trace({"root_stage_world_xyz_m": trace,
                                      "published_frame_count": 7},
                                    "root_stage_world_xyz_m")
    assert mismatch["status"] == "published_frame_count_mismatch"
    assert mismatch["values"] is None


def test_heightfield_knownpoints_and_sampler_do_not_clip_outside_or_fill_unknowns():
    sampler, summary = probe.TerrainHeightfield.from_config({"terrain_scene": {
        "hf": [[1.0, None], [3.0, 4.0]], "min_point": [10.0, 50.0, 20.0],
        "dx": 1.0, "heightfield_valid": [[True, True], [True, False]],
    }})
    assert sampler is not None
    assert summary["status"] == "valid" and summary["known_points"] == 2
    assert summary["total_points"] == 4
    assert sampler.sample(10.0, 20.0) == (1.0, "valid")
    assert sampler.sample(11.0, 20.0) == (None, "unknown_heightfield_cell")
    assert sampler.sample(12.0, 20.0) == (None, "outside_heightfield")
    invalid_sampler, invalid_summary = probe.TerrainHeightfield.from_config({
        "terrain_scene": {"hf": [[None]], "min_point": [0, 0, 0], "dx": 1.0}})
    assert invalid_sampler is None and invalid_summary["status"] == "invalid"
    absent_sampler, absent_summary = probe.TerrainHeightfield.from_config({})
    assert absent_sampler is None and absent_summary["status"] == "missing"


def test_timing_summaries_separate_phase_and_neural_from_cached_and_stay_finite():
    rows = [
        {"phase": "idle", "generation_status": "neural", "response_ms": 10.0,
         "inference_ms": 8.0, "skin_ms": 2.0},
        {"phase": "straight", "generation_status": "cached", "response_ms": 3.0,
         "inference_ms": 0.0, "skin_ms": 2.0},
        {"phase": "straight", "generation_status": "unknown", "response_ms": float("nan"),
         "inference_ms": None, "skin_ms": None},
    ]
    result = probe.summarize_timings(rows)
    assert result["by_phase"]["idle"]["response_ms"]["mean"] == 10.0
    assert result["by_generation"]["cached"]["inference_ms"]["mean"] == 0.0
    assert result["by_generation"]["unknown"]["response_ms"]["count"] == 0
    json.dumps(result, allow_nan=False)


def test_playback_pacing_uses_only_the_published_block_duration():
    assert probe.PUBLISH_FRAMES == 8
    assert probe.PUBLISH_FRAMES / 30 == pytest.approx(8 / 30)
    assert probe.build_parser().get_default("no_pace") is False


def test_websocket_probe_advances_only_after_full_ack_and_emits_finite_json():
    async def scenario():
        state = {"commands": [], "acks": [], "pending": False, "root": [0.0, 0.8, 0.0]}

        async def config(_request):
            return web.json_response({"terrain_scene": {
                "hf": [[0.0, 0.0], [0.0, 0.0]],
                "min_point": [-1.0, 0.0, -1.0], "dx": 1.0,
            }})

        async def websocket(request):
            socket = web.WebSocketResponse(heartbeat=20)
            await socket.prepare(request)
            await socket.send_json({"type": "ready", "mode": "live", "fps": 30,
                                    "vertex_count": 1, "root": state["root"]})
            async for message in socket:
                command = json.loads(message.data)
                if command["type"] == "reset":
                    assert command["mode"] == "terrain"
                    await socket.send_json({"type": "ready", "mode": "terrain", "fps": 30,
                                            "vertex_count": 1, "root": state["root"]})
                elif command["type"] == "plan":
                    assert not state["pending"], "next plan arrived before previous ACK"
                    state["pending"] = True
                    state["commands"].append(command)
                    roots = np.asarray([
                        [state["root"][0] + 0.01 * (index + 1), 0.8, state["root"][2]]
                        for index in range(8)], dtype=np.float32)
                    debug = {
                        "published_frame_count": 8,
                        "observation_root_world_xyz_m": list(state["root"]),
                        "root_stage_world_xyz_m": np.column_stack((roots[:, 0],
                            roots[:, 1] + 0.4, roots[:, 2])).tolist(),
                        "decoded_world_xyz_m": np.column_stack((roots[:, 0],
                            roots[:, 1] + 0.05, roots[:, 2])).tolist(),
                    }
                    request_id = command["request_id"]
                    await socket.send_json({
                        "type": "plan", "request_id": request_id, "mode": "terrain",
                        "frame_count": 8, "vertex_count": 1, "frame_stride_floats": 6,
                        "encoding": "float32-le", "inference_ms": 5.0 if request_id == 1 else 0.0,
                        "skin_ms": 1.5, "generation": {"neural_generation": request_id == 1},
                        "debug": debug,
                    })
                    binary_rows = np.zeros((8, 6), dtype="<f4")
                    binary_rows[:, :3] = roots
                    await socket.send_bytes(binary_rows.tobytes())
                elif command["type"] == "played":
                    state["acks"].append(command)
                    assert state["pending"]
                    assert command["frame_count"] == 8
                    state["pending"] = False
                    state["root"] = [
                        float(state["root"][0] + 0.08), 0.8, float(state["root"][2])]
                    await socket.send_json({"type": "played", "request_id": command["request_id"],
                                            "frame_count": 8})
            return socket

        app = web.Application()
        app.router.add_get("/api/config", config)
        app.router.add_get("/ws", websocket)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            report = await probe.run_probe(f"http://127.0.0.1:{port}", blocks=7,
                                           speed_mps=1.25, paced=False)
        finally:
            await runner.cleanup()
        return report, state

    report, state = asyncio.run(scenario())
    assert report["status"] == "complete"
    assert report["execution_mode"] == "offline_accelerated_websocket"
    assert report["offline_accelerated"] is True
    assert [row["phase"] for row in report["rows"]] == ["idle"] * 6 + ["straight"]
    assert state["commands"][-1]["velocity_xz"] == [0.0, 1.25]
    assert len(state["acks"]) == 7 and all(ack["frame_count"] == 8 for ack in state["acks"])
    assert report["summary"]["acknowledged_blocks"] == 7
    assert report["terrain_knownpoints"]["known_points"] == 4
    assert report["rows"][0]["current_obs_root_y_m"] == pytest.approx(0.8)
    assert report["rows"][0]["predicted_root_y_m"] == pytest.approx([1.2] * 8)
    assert report["rows"][0]["final_decoded_y_m"] == pytest.approx([0.85] * 8)
    assert report["rows"][1]["boundary_displacement_m"] == pytest.approx(0.01)
    assert report["rows"][0]["frames"][0]["ground_root_gap_y_m"] == pytest.approx(0.8)
    assert report["summary"]["displacements"]["internal_frame_displacement_m"]["idle"]["count"] == 42
    assert report["motion_quality_assessed"] is False
    json.dumps(report, allow_nan=False)
