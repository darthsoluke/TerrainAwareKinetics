"""Small causal-command tests plus opt-in generated-artifact checks."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_parc_local_terrain_pilot.py"
SPEC = importlib.util.spec_from_file_location("parc_local_terrain_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class PilotUnitTests(unittest.TestCase):
    def test_scripted_intent_uses_only_current_heading(self):
        local, world = pilot.transform_local_intent((3.0, 0.0), 1.5707963267948966)
        self.assertAlmostEqual(local[0], 1.0)
        self.assertAlmostEqual(local[1], 0.0)
        self.assertAlmostEqual(world[0], 0.0, places=7)
        self.assertAlmostEqual(world[1], 1.0, places=7)

    def test_zero_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.transform_local_intent((0.0, 0.0), 0.0)

    def test_first_current_contact_requires_history(self):
        self.assertEqual(
            pilot.first_contact_with_history([True, False, True], previous_frame_count=2),
            2,
        )
        with self.assertRaises(ValueError):
            pilot.first_contact_with_history([True, False], previous_frame_count=2)

    def test_motion_metadata_never_claims_physics_correction(self):
        metadata = pilot.motion_metadata("scripted local continuation")
        self.assertEqual(metadata["motion_origin"], "PARC official generator raw kinematic output")
        self.assertFalse(metadata["physics_corrected"])

    def test_each_output_run_gets_a_new_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = pilot.unique_run_directory(root)
            second = pilot.unique_run_directory(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


@unittest.skipUnless(os.environ.get("PARC_PILOT_DIR"), "set PARC_PILOT_DIR to opt in to generated-artifact checks")
class ParcPilotArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot_dir = Path(os.environ["PARC_PILOT_DIR"])

    def test_report_and_raw_outputs_are_present(self):
        report_path = self.pilot_dir / "pilot_report.json"
        self.assertTrue(report_path.is_file(), "PARC_PILOT_DIR must name one completed run directory")
        report = json.loads(report_path.read_text())
        self.assertIn(report["status"], {"complete", "partial"})
        self.assertEqual(report["motion_kind"], "raw_kinematic")
        self.assertFalse(report["physics_corrected"])
        self.assertEqual(report["gpu_used"], False)
        self.assertEqual(report["model_config"]["local_observation_shape"], [31, 31])
        self.assertEqual(report["coordinate_frame"]["soma30_retarget"], "not approved or performed")

        cases = {item["case"]: item for item in report["cases"]}
        stairs = {name for name in ("stairs_up", "stairs_down") if name in cases}
        if stairs:
            self.assertEqual(stairs, {"stairs_up", "stairs_down"})
            for name in stairs:
                case_dir = self.pilot_dir / name
                self.assertTrue((case_dir / "generated_raw_kinematic.pkl").is_file())
                self.assertTrue((case_dir / "terrain_and_route.npz").is_file())
                self.assertTrue((case_dir / "local_model_observations.npz").is_file())
                self.assertTrue((case_dir / "kinematic_world_terrain_diagnostics.npz").is_file())
                self.assertEqual(cases[name]["motion_kind"], "raw_kinematic")
                self.assertFalse(cases[name]["physics_corrected"])

        if "hand_contact_local_continuation" in cases:
            item = cases["hand_contact_local_continuation"]
            self.assertEqual(item["status"], "generation_complete", item.get("failure_reason"))
            self.assertTrue(item["behavior_success_not_assessed"])
            self.assertIn("scripted_intent_local_xy", item)
            self.assertNotIn("future_idx", item)
            self.assertTrue((self.pilot_dir / "hand_contact_continuation" / "generated_raw_kinematic.pkl").is_file())
        elif "hand_contact_local_continuation" in report.get("requested_cases", []):
            self.fail("requested hand-contact continuation is absent from the report")

    def test_hand_contact_continuation_is_not_scored_as_behavior_success(self):
        report = json.loads((self.pilot_dir / "pilot_report.json").read_text())
        cases = {item["case"]: item for item in report["cases"]}
        if "hand_contact_local_continuation" not in cases:
            requested = report.get("requested_cases")
            if requested is None or "hand_contact_local_continuation" in requested:
                self.fail("requested hand-contact continuation is absent from the report")
            self.skipTest("hand-contact continuation was explicitly not requested")
        item = cases["hand_contact_local_continuation"]
        self.assertEqual(item["status"], "generation_complete", item.get("failure_reason"))
        self.assertTrue(item["behavior_success_not_assessed"])
        self.assertGreaterEqual(item["generated_hand_contact_frames"], 0)
        self.assertEqual(item["conditioning_observation_shape"], [31, 31])
        self.assertIn("scripted_intent_local_xy", item)
        self.assertNotIn("future_idx", item)
        self.assertTrue((self.pilot_dir / "hand_contact_continuation" / "generated_raw_kinematic.pkl").is_file())
        self.assertTrue((self.pilot_dir / "hand_contact_continuation" / "source_kinematic_world_terrain_diagnostics.npz").is_file())


if __name__ == "__main__":
    unittest.main()
