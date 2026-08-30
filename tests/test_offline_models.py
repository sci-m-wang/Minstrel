import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_offline_models", ROOT / "scripts/download_offline_models.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OfflineModelSourceTests(unittest.TestCase):
    def test_llama_uses_pinned_modelscope_without_runtime_knobs(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        item = registry["models"]["llama-3.1-8b-instruct"]
        command = MODULE.download_command(item, Path("/models/llama"))
        self.assertEqual(command[:3], ["modelscope", "download", "--model"])
        self.assertIn("LLM-Research/Meta-Llama-3.1-8B-Instruct", command)
        self.assertIn("master", command)
        self.assertIn("original/*", command)
        self.assertNotIn("hf", command)
        self.assertNotIn("--max-workers", command)
        self.assertNotIn("--max-retries", command)

    def test_modelscope_revision_is_separately_pinned_to_exact_commit(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        item = registry["models"]["llama-3.1-8b-instruct"]
        self.assertEqual(item["source_revision"], "master")
        self.assertEqual(item["revision"], "359efdbb8af05b788a4ad4185215c6b8caa9052c")

    def test_other_models_keep_declared_huggingface_source(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        item = registry["models"]["qwen2.5-7b-instruct"]
        command = MODULE.download_command(item, Path("/models/qwen"))
        self.assertEqual(command[:2], ["hf", "download"])


if __name__ == "__main__":
    unittest.main()
