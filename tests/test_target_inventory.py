import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inspect_target",
    ROOT / "skills/side-profile-experiment/scripts/inspect_target.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TargetInventoryTests(unittest.TestCase):
    def test_non_executable_nvidia_smi_is_recorded_not_raised(self):
        with mock.patch.object(MODULE.shutil, "which", return_value="/bad/nvidia-smi"):
            with mock.patch.object(MODULE.subprocess, "run", side_effect=OSError("not executable")):
                result = MODULE.visible_gpus()
        self.assertFalse(result["visible"])
        self.assertEqual(result["devices"], [])
        self.assertIn("not executable", result["detail"])


if __name__ == "__main__":
    unittest.main()
