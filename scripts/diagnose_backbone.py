import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModel

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[1]
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # If the model is cached in the default home directory, copy it to the local workspace cache
    default_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if default_cache.exists():
        for p in default_cache.glob("models--facebook--dinov3*"):
            if p.is_dir():
                target_hub_dir = hf_cache_dir / "hub"
                target_dir = target_hub_dir / p.name
                if not target_dir.exists():
                    try:
                        import shutil
                        target_hub_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(p, target_dir, symlinks=True)
                    except Exception:
                        pass


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
