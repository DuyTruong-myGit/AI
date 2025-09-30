# src/eval.py
import os, json, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from dataset import DermDataset
from model_factory import create_model, DEFAULT_ARCH

ARTI = "artifacts/models"
MODEL = os.path.join(ARTI,"best_timm.pt")   # đổi tên mới
OUT = "artifacts/metrics"
os.makedirs(OUT, exist_ok=True)

@torch.no_grad()
def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(MODEL, map_location=device)
    arch = obj.get("arch", DEFAULT_ARCH)
    idx2 = obj.get("idx2label", None)
    if idx2 is None:
        raise ValueError("Checkpoint không có idx2label.")
    model = create_model(arch, num_classes=len(idx2), pretrained=False).to(device).eval()

    from collections import OrderedDict
    new_sd = OrderedDict((k.replace("module.",""), v) for k,v in obj["state_dict"].items())
    model.load_state_dict(new_sd, strict=True)

    ds = DermDataset("data/splits/test.csv", is_train=False)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

    y_true, y_pred, y_score = [], [], []
    for x,y in ld:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_score.append(prob)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(prob.argmax(1).tolist())

    y_score = np.concatenate(y_score, axis=0)
    rep = classification_report(y_true, y_pred, target_names=idx2, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred)
    try:
        auc_macro = roc_auc_score(np.eye(len(idx2))[y_true], y_score, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = None

    with open(os.path.join(OUT,"report.json"),"w") as f:
        json.dump({"classification_report":rep,"confusion_matrix":cm.tolist(),"roc_auc_macro":auc_macro}, f, indent=2)
    print("Saved metrics to artifacts/metrics/report.json")
    print("Macro-F1:", rep["macro avg"]["f1-score"], "| ROC-AUC(macro):", auc_macro)

if __name__ == "__main__":
    evaluate()
