# src/infer_torch.py
import os, json, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model_factory import create_model, DEFAULT_ARCH

# Mapping nhãn
IDX2LABEL_PATH = "artifacts/models/idx2label.json"
IDX2LABEL = json.load(open(IDX2LABEL_PATH, "r", encoding="utf-8")) if os.path.exists(IDX2LABEL_PATH) else None

# Tiền xử lý (224x224, ImageNet mean/std)
IMG_SIZE = 224
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
_preprocess = transforms.Compose([
    ImageOps.exif_transpose,
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def load_model(weights_path="artifacts/models/best_timm.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(weights_path, map_location=device)  # dict: state_dict, arch, idx2label
    arch = obj.get("arch", DEFAULT_ARCH)
    idx2 = obj.get("idx2label", IDX2LABEL)
    if idx2 is None:
        raise ValueError("Thiếu idx2label. Hãy train/lưu checkpoint với idx2label.")
    model = create_model(arch, num_classes=len(idx2), pretrained=False).to(device).eval()

    from collections import OrderedDict
    new_sd = OrderedDict((k.replace("module.",""), v) for k,v in obj["state_dict"].items())
    model.load_state_dict(new_sd, strict=True)
    return model, device, idx2

@torch.no_grad()
def infer_image(model, device, pil_image, idx2label=None, topk=5, reject_threshold=0.8):
    x = _preprocess(pil_image).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1][:topk]
    idx2 = idx2label or IDX2LABEL
    top = [{"label": (idx2[i] if idx2 else int(i)), "prob": float(probs[i])} for i in order]
    maxp = float(probs[order[0]])
    return {
        "top_label": top[0]["label"] if maxp >= reject_threshold else "unknown",
        "predictions": top,
        "unsure": maxp < reject_threshold,
        "max_prob": maxp,
        "margin": maxp - float(probs[order[1]]) if len(order) > 1 else maxp
    }
