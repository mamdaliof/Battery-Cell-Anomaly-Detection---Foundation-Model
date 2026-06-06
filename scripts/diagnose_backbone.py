import argparse
import torch
from transformers import AutoModel

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
