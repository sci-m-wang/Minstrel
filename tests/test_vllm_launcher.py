import sys
import types
from unittest import mock

from sideprofile import vllm_launcher


def test_vllm_launcher_isolates_optional_vision_before_cli():
    state = {"isolated": False, "called": False}

    def isolate():
        state["isolated"] = True

    def fake_main():
        assert state["isolated"]
        state["called"] = True
        return 0

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.cli": types.ModuleType("vllm.entrypoints.cli"),
        "vllm.entrypoints.cli.main": types.ModuleType("vllm.entrypoints.cli.main"),
    }
    modules["vllm.entrypoints.cli.main"].main = fake_main
    with mock.patch.dict(sys.modules, modules):
        with mock.patch.object(vllm_launcher, "isolate_text_transformers_runtime", isolate):
            assert vllm_launcher.main() == 0
    assert state["called"]
