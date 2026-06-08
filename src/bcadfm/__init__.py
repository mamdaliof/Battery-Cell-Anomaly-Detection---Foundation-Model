import os
from pathlib import Path

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[2]
    hf_cache_dir = workspace_root / "models" / "hf_cache"

    # Check if the model is cached in the default home directory
    default_cache = Path.home() / ".cache" / "huggingface" / "hub"
    has_dinov3_default = False
    if default_cache.exists():
        for p in default_cache.glob("models--facebook--dinov3*"):
            if p.is_dir():
                has_dinov3_default = True
                break

    workspace_cache_hub = hf_cache_dir / "hub"
    has_dinov3_workspace = False
    if workspace_cache_hub.exists():
        for p in workspace_cache_hub.glob("models--facebook--dinov3*"):
            if p.is_dir():
                has_dinov3_workspace = True
                break

    # If it is in the default cache but not in the workspace cache,
    # do NOT override HF_HOME (allow using the default cache).
    if has_dinov3_workspace or not has_dinov3_default:
        os.environ["HF_HOME"] = str(hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("bcadfm")
except PackageNotFoundError:  # local, editable install during development
    __version__ = "0.0.0"

