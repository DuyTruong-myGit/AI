import os
import json
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFile
from torchvision import transforms

# Cho phép đọc ảnh bị truncate một phần (một số ảnh tải về lỗi)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----- ĐƯỜNG DẪN MAPPING LỚP -----
IDX2LABEL_PATHS = [
    os.path.join("artifacts", "models", "idx2label.json"),
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "models", "idx2label.json"),
]
IDX2LABEL = None
for p in IDX2LABEL_PATHS:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            IDX2LABEL = json.load(f)
        break
if IDX2LABEL is None:
    raise FileNotFoundError(
        "Không tìm thấy artifacts/models/idx2label.json. Hãy chạy prepare_data.py trước."
    )
LABEL2IDX = {c: i for i, c in enumerate(IDX2LABEL)}

# ----- HẰNG SỐ TIỀN XỬ LÝ -----
IMG_SIZE: int = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ----- HÀM CHUYỂN ĐỔI ẢNH (top-level -> picklable) -----
def fix_exif(im: Image.Image) -> Image.Image:
    # Sửa xoay EXIF cho ảnh từ điện thoại
    return ImageOps.exif_transpose(im)

def to_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")

# ----- TRANSFORMS -----
train_tfms = transforms.Compose([
    transforms.Lambda(fix_exif),
    transforms.Lambda(to_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.10, 0.05)], p=0.6),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tfms = transforms.Compose([
    transforms.Lambda(fix_exif),
    transforms.Lambda(to_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class DermDataset(Dataset):
    """
    CSV format yêu cầu: 2 cột
      - image_path: đường dẫn tuyệt đối/ tương đối đến file ảnh
      - label     : mã lớp (ví dụ: nv, mel, bkl, ...)
    """
    def __init__(self, csv_path: str, is_train: bool = True):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không thấy CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        missing_cols = {"image_path", "label"} - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Thiếu cột trong CSV {csv_path}: {missing_cols}")
        self.tfms = train_tfms if is_train else val_tfms

        # (Tuỳ chọn) lọc dòng bị thiếu file
        self.df = self.df[self.df["image_path"].apply(os.path.exists)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"CSV {csv_path} không còn mẫu hợp lệ sau khi lọc đường dẫn ảnh.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label_str = row["label"]

        try:
            img = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Lỗi mở ảnh: {img_path} -> {e}")

        x = self.tfms(img)
        if label_str not in LABEL2IDX:
            raise KeyError(f"Nhãn '{label_str}' không có trong mapping IDX2LABEL: {IDX2LABEL}")
        y = LABEL2IDX[label_str]
        return x, torch.tensor(y, dtype=torch.long)
