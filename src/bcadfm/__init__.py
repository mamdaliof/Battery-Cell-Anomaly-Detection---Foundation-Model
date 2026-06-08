import os
from pathlib import Path

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[2]
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)
    # Ensure the directory exists
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("bcadfm")
except PackageNotFoundError:  # local, editable install during development
    __version__ = "0.0.0"

