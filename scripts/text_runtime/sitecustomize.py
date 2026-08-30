"""Keep vLLM's complete Python process tree on the text-only Transformers path."""

try:
    import transformers.utils.import_utils as transformer_imports
except ImportError:
    transformer_imports = None

if transformer_imports is not None:
    transformer_imports._torchvision_available = False
