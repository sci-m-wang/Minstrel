import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_roleagentbench.py"
SPEC = importlib.util.spec_from_file_location("roleagent_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_selected_choice_accepts_label_or_verbatim_choice() -> None:
    choices = ["A. First answer", "B. Second answer"]
    assert MODULE.selected_choice("B. Second answer", choices) == "B"
    assert MODULE.selected_choice("I would choose A because it fits.", choices) == "A"
    assert MODULE.selected_choice("The second answer is Second answer", choices) == "B"
