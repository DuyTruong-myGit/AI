# src/sanity_check.py
import json, torch
from model_factory import create_model, DEFAULT_ARCH

def main():
    obj = torch.load("artifacts/models/best_timm.pt", map_location="cpu")
    arch = obj.get("arch", DEFAULT_ARCH)
    idx2 = obj.get("idx2label", [])
    print({
        "arch": arch,
        "num_labels": len(idx2),
        "labels": idx2
    })

if __name__ == "__main__":
    main()
