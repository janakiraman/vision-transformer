"""
Vision Transformer inference API.
Supports two model types, selected via MODEL_TYPE env var:
  - "huggingface" : transformers ViTForImageClassification (default)
  - "pytorch"     : torchvision vit_b_16 loaded from a state_dict (.pth)
The model is baked into the image at /app/model during docker build.
"""
import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_TYPE = os.getenv("MODEL_TYPE", "huggingface").lower()
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="ViT Inference API")

model = None
processor = None          # HF image processor
transform = None          # torchvision transforms
labels = None             # class-id -> name


def load_huggingface():
    """Load a HF ViT model + processor from MODEL_DIR."""
    global model, processor, labels
    from transformers import ViTForImageClassification, ViTImageProcessor

    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    labels = model.config.id2label
    model.to(DEVICE).eval()


def load_pytorch():
    """Load torchvision vit_b_16 with a custom state_dict (model.pth)."""
    global model, transform, labels
    import json

    from torchvision import transforms
    from torchvision.models import vit_b_16

    num_classes = int(os.getenv("NUM_CLASSES", "1000"))
    model = vit_b_16(num_classes=num_classes)
    state = torch.load(
        os.path.join(MODEL_DIR, "model.pth"),
        map_location=DEVICE, weights_only=True,
    )
    # handle checkpoints saved as {"state_dict": ...}
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    labels_file = os.path.join(MODEL_DIR, "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            labels = {int(k): v for k, v in json.load(f).items()}
    else:
        labels = {i: str(i) for i in range(num_classes)}


@app.on_event("startup")
def startup():
    if MODEL_TYPE == "pytorch":
        load_pytorch()
    else:
        load_huggingface()
    print(f"Loaded {MODEL_TYPE} model from {MODEL_DIR} on {DEVICE}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_type": MODEL_TYPE, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    with torch.no_grad():
        if MODEL_TYPE == "pytorch":
            x = transform(image).unsqueeze(0).to(DEVICE)
            logits = model(x)
        else:
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
    return {
        "predictions": [
            {"label": labels[i.item()], "class_id": i.item(),
             "score": round(p.item(), 4)}
            for p, i in zip(top.values, top.indices)
        ]
    }
