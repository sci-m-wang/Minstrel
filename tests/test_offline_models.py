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

    def test_executable_actor_matrix_excludes_gemma(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("gemma-2-9b-it", registry["models"])
        self.assertNotIn("gemma-2-9b-it", registry["actor_matrix"]["panel_a"])
        self.assertEqual(
            registry["actor_matrix"]["panel_a"],
            [
                "llama-3.1-8b-instruct",
                "qwen2.5-7b-instruct",
                "qwen2.5-14b-instruct",
                "mistral-7b-instruct-v0.3",
            ],
        )

    def test_profiler_is_connected_gpt_and_not_a_local_model_asset(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            registry["profiler"],
            {
                "provider": "GPT",
                "model": "gpt-5.6-sol",
                "execution_location": "connected_preparation",
            },
        )
        self.assertNotIn("gpt-5.6-sol", registry["models"])

    def test_retrieval_services_are_connected_and_not_local_assets(self):
        registry = yaml.safe_load((ROOT / "offline/models.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            registry["retrieval_preparation"],
            {
                "embedding": {
                    "provider": "GPT",
                    "model": "text-embedding-3-small",
                    "execution_location": "connected_preparation",
                },
                "reranker": {
                    "provider": "COHERE",
                    "model": "Cohere-rerank-v4.0-pro",
                    "execution_location": "connected_preparation",
                },
            },
        )
        self.assertNotIn("qwen3-embedding-0.6b", registry["models"])
        self.assertNotIn("bge-reranker-v2-m3", registry["models"])


if __name__ == "__main__":
    unittest.main()
