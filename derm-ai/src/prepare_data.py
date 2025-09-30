# src/prepare_data.py
import os, json, re, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_DIR = "data/raw"
CSV = os.path.join(RAW_DIR, "labels.csv")
OUT_DIR = "data/splits"
ARTI_DIR = "artifacts/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTI_DIR, exist_ok=True)

# Thứ tự lớp CHUẨN của HAM10000
CANONICAL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# =============== Helpers ===============
def norm_text(x: str) -> str:
    if pd.isna(x): return ""
    x = x.lower()
    x = re.sub(r"[\(\),\-_/]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def contains(txt: str, keys) -> bool:
    return any(k in txt for k in keys)

def robust_map_to_7(row: pd.Series) -> str | None:
    """
    Map diagnosis_1..3 (+ một vài cột hỗ trợ) -> 1 trong 7 lớp:
    akiec, bcc, bkl, df, mel, nv, vasc
    """
    d1 = norm_text(row.get("diagnosis_1", ""))
    d2 = norm_text(row.get("diagnosis_2", ""))
    d3 = norm_text(row.get("diagnosis_3", ""))
    combo = f"{d1} {d2} {d3}"

    # Ưu tiên từ khóa đặc trưng
    if contains(combo, ["melanoma", "malignant melanocytic"]):
        return "mel"

    if contains(combo, ["basal cell carcinoma", "bcc"]):
        return "bcc"

    if contains(combo, ["actinic keratosis", "bowen", "akiec"]):
        return "akiec"

    if contains(combo, [
        "keratosis", "keratoses", "keratosis like",
        "seborrheic", "solar lentigo", "lentigo",
        "lichen planus"
    ]):
        # HAM10000 gộp nhóm "benign keratosis-like lesions"
        return "bkl"

    if contains(combo, ["dermatofibroma", "df"]):
        return "df"

    if contains(combo, ["vascular", "hemangioma", "angioma", "angiokeratoma", "vasc"]):
        return "vasc"

    if contains(combo, ["nevus", "naevus", "nevi", "melanocytic nevi", "benign melanocytic"]):
        return "nv"

    # Fallback bằng cột hỗ trợ (nếu có)
    melanocytic = str(row.get("melanocytic", "")).strip().lower()
    if melanocytic in ["true", "1"]:
        if "benign" in combo:
            return "nv"
        if "malignant" in combo:
            return "mel"

    return None  # Không map được

def build_image_path(isic_id: str) -> str:
    # Ảnh ISIC đa số .jpg; nếu cần, có thể thêm .jpeg/.png fallback
    jpg = os.path.join(RAW_DIR, f"{isic_id}.jpg")
    if os.path.exists(jpg):
        return jpg
    jpeg = os.path.join(RAW_DIR, f"{isic_id}.jpeg")
    if os.path.exists(jpeg):
        return jpeg
    png = os.path.join(RAW_DIR, f"{isic_id}.png")
    if os.path.exists(png):
        return png
    return jpg  # để lọc not exists bước sau

# =============== Load & map ===============
df = pd.read_csv(CSV)
if "isic_id" not in df.columns:
    raise RuntimeError("labels.csv thiếu cột 'isic_id'.")

df["label"] = df.apply(robust_map_to_7, axis=1)
df = df.dropna(subset=["label"]).copy()

# Đường dẫn ảnh
df["image_path"] = df["isic_id"].apply(build_image_path)
df = df[df["image_path"].apply(os.path.exists)].copy()

# Group theo lesion_id để tránh leakage; nếu thiếu thì fallback isic_id
if "lesion_id" in df.columns:
    df["group"] = df["lesion_id"].fillna(df["isic_id"])
else:
    df["group"] = df["isic_id"]

# Giữ các cột cần thiết
df = df[["image_path", "label", "group"]]

# =============== Chọn thứ tự nhãn và lưu idx2label.json ===============
present = sorted(df["label"].unique().tolist(), key=lambda x: CANONICAL_CLASSES.index(x))
missing = [c for c in CANONICAL_CLASSES if c not in present]

if missing:
    print(f"[!] Cảnh báo: dataset hiện thiếu các lớp: {missing}")
    # Mặc định: chỉ dùng các lớp hiện có để tránh out_dim lệch
    idx2label = present
else:
    idx2label = CANONICAL_CLASSES

with open(os.path.join(ARTI_DIR, "idx2label.json"), "w", encoding="utf-8") as f:
    json.dump(idx2label, f, ensure_ascii=False, indent=2)

print("[i] Lớp sẽ dùng để train:", idx2label)

# =============== Split theo group (lesion-aware) ===============
# 70% train, 15% val, 15% test
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df["group"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df  = df.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group"]))
val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

# (Tuỳ chọn) lọc theo idx2label để chắc chắn không lẫn lớp missing
train_df = train_df[train_df["label"].isin(idx2label)].reset_index(drop=True)
val_df   = val_df[val_df["label"].isin(idx2label)].reset_index(drop=True)
test_df  = test_df[test_df["label"].isin(idx2label)].reset_index(drop=True)

# =============== Lưu CSV ===============
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

# =============== Thống kê nhanh ===============
def counts(df_):
    return df_.groupby("label").size().sort_values(ascending=False).to_dict()

print("[count] train:", counts(train_df))
print("[count] val  :", counts(val_df))
print("[count] test :", counts(test_df))
print("[done] Wrote:",
      os.path.join(OUT_DIR,"train.csv"),
      os.path.join(OUT_DIR,"val.csv"),
      os.path.join(OUT_DIR,"test.csv"))
