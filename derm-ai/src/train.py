# src/train.py
import os, time, json, random, platform, warnings
from typing import Tuple
from torch import amp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import DermDataset, IDX2LABEL
from model_factory import create_model, DEFAULT_ARCH

# ========= CẤU HÌNH =========
ARCH: str = DEFAULT_ARCH         # đổi backbone ở đây khi muốn
EPOCHS: int = 5                 # tăng nhẹ để khai thác ConvNeXt
BATCH_SIZE: int = 32
LEARNING_RATE: float = 2e-3
WEIGHT_DECAY: float = 1e-4
GRAD_ACCUM_STEPS: int = 1
MIXED_PRECISION: bool = True     # RTX 3050: nên bật
SAVE_DIR: str = os.path.join("artifacts", "models")
BEST_PATH: str = os.path.join(SAVE_DIR, "best_timm.pt")

TRAIN_CSV: str = os.path.join("data", "splits", "train.csv")
VAL_CSV:   str = os.path.join("data", "splits", "val.csv")
SEED: int = 42

# ========= TIỆN ÍCH =========
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # cho GPU nhanh hơn

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def suggest_num_workers() -> int:
    if platform.system().lower().startswith("win"):
        return 0  # an toàn cho Windows
    try:
        import multiprocessing as mp
        return max(2, min(4, mp.cpu_count() - 1))
    except Exception:
        return 2

# ========= VÒNG TRAIN/VAL =========
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: str,
                    scaler: torch.cuda.amp.GradScaler | None = None) -> Tuple[float, float, float]:
    model.train()
    total_loss = total = correct = 0
    all_y, all_p = [], []
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if scaler is not None:
            with amp.autocast('cuda', enabled=(device == "cuda" and MIXED_PRECISION)):
                logits = model(x)
                loss = criterion(logits, y) / GRAD_ACCUM_STEPS
        else:
            logits = model(x)
            loss = criterion(logits, y) / GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            total_loss += (loss.item() * x.size(0) * GRAD_ACCUM_STEPS)
            all_y.extend(y.detach().cpu().tolist())
            all_p.extend(preds.detach().cpu().tolist())

    acc = correct / total if total else 0.0
    f1 = f1_score(all_y, all_p, average="macro") if all_y else 0.0
    return total_loss / max(total, 1), acc, f1

@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: str) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total = correct = 0
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += x.size(0)
        total_loss += loss.item() * x.size(0)
        all_y.extend(y.cpu().tolist()); all_p.extend(preds.cpu().tolist())
    acc = correct / total if total else 0.0
    f1 = f1_score(all_y, all_p, average="macro") if all_y else 0.0
    return total_loss / max(total, 1), acc, f1

# ========= MAIN =========
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)

    device = get_device()
    num_classes = len(IDX2LABEL)
    print(f"Device: {device} | Arch: {ARCH} | Classes: {IDX2LABEL}")

    num_workers = suggest_num_workers()
    pin_memory = (device == "cuda")

    train_ld = DataLoader(DermDataset(TRAIN_CSV, is_train=True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))
    val_ld   = DataLoader(DermDataset(VAL_CSV,   is_train=False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))

    # ConvNeXt-Tiny (timm)
    model = create_model(ARCH, num_classes=num_classes, pretrained=True).to(device)
    # Label smoothing giúp ổn định
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    scaler = amp.GradScaler('cuda', enabled=(device == "cuda" and MIXED_PRECISION))

    best_f1 = -1.0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_ld, optimizer, criterion, device, scaler)
        va_loss, va_acc, va_f1 = validate(model, val_ld, criterion, device)
        scheduler.step()
        print(f"[{ep:02d}] train {tr_loss:.4f}/{tr_acc:.3f}/{tr_f1:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f}/{va_f1:.3f} | {time.time()-t0:.1f}s")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "state_dict": model.state_dict(),
                "arch": ARCH,
                "idx2label": IDX2LABEL
            }, BEST_PATH)
            print(f"  -> saved: {BEST_PATH} f1: {best_f1:.3f}")

    summary = {
        "best_f1": best_f1, "classes": IDX2LABEL, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "grad_accum_steps": GRAD_ACCUM_STEPS, "mixed_precision": bool(scaler.is_enabled()) if scaler else False,
        "device": device, "arch": ARCH
    }
    with open(os.path.join(SAVE_DIR, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary ->", os.path.join(SAVE_DIR, "train_summary.json"))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
    main()
