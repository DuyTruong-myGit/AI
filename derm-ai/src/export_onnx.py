# src/export_onnx.py
import torch, os, json
from dataset import IMG_SIZE
from model_factory import create_model, DEFAULT_ARCH

ARTI = "artifacts/models"
MODEL_PT = os.path.join(ARTI,"best_timm.pt")
OUT_ONNX = os.path.join(ARTI,"model.onnx")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(MODEL_PT, map_location=device)
    arch = obj.get("arch", DEFAULT_ARCH)
    idx2 = obj.get("idx2label", None)
    assert idx2 is not None, "Checkpoint thiếu idx2label."

    model = create_model(arch, num_classes=len(idx2), pretrained=False).to(device).eval()
    from collections import OrderedDict
    new_sd = OrderedDict((k.replace("module.",""), v) for k,v in obj["state_dict"].items())
    model.load_state_dict(new_sd, strict=True)

    dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE, device=device)
    torch.onnx.export(model, dummy, OUT_ONNX,
        input_names=["input"], output_names=["logits"],
        opset_version=13, dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print("Exported ONNX ->", OUT_ONNX)

if __name__ == "__main__":
    main()
