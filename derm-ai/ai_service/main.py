from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.infer_torch import load_model, infer_image

app = FastAPI(title="Derm AI Inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model, device = load_model("artifacts/models/best_resnet18.pt")

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw))
    result = infer_image(model, device, pil, topk=5)
    return result
