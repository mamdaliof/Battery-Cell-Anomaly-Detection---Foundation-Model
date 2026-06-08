import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModel

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[1]
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


def diagnose(model_name: str):
    print(f"\n==========================================")
    print(f"🔍 DIAGNOSING MODULES FOR: {model_name}")
    print(f"==========================================\n")
    
    try:
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("--- Linear Modules ---")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  - {name}")
            count += 1
            if count >= 30:
                print("  ... (truncated)")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k")
    args = parser.parse_args()
    diagnose(args.model)
