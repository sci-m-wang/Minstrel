from __future__ import annotations

import sys


def isolate_text_transformers_runtime() -> None:
    """Keep optional vision extensions out of this text-only vLLM process."""

    import transformers.utils.import_utils as transformer_imports

    transformer_imports._torchvision_available = False


def main() -> int | None:
    """Launch vLLM for this text-only experiment without optional vision extensions."""

    isolate_text_transformers_runtime()
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv[0] = "vllm"
    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
