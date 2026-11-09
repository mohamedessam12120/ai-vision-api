import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageStat, UnidentifiedImageError
from torchvision import models, transforms

MODEL_PATH = Path("models/resnet18_pets.pt")
CLASSES_PATH = Path("models/classes.json")
MAX_BYTES = 5_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png"}
UNCERTAIN_THRESHOLD = 0.6
LOG_PATH = Path("logs/predictions.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# Same preprocessing as evaluation in train.py
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = json.loads(CLASSES_PATH.read_text())
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    state.update(model=model, classes=classes, device=device)
    log.info("model loaded on %s with %d classes", device, len(classes))
    yield
    state.clear()


app = FastAPI(title="Pets Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(state["device"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Input validation
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")

    start = time.perf_counter()
    x = preprocess(image).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        probs = torch.softmax(state["model"](x), dim=1)[0]
    top = torch.topk(probs, 3)
    results = [
        {"class": state["classes"][i], "confidence": round(p, 4)}
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    ms = round((time.perf_counter() - start) * 1000, 1)

    confidence = results[0]["confidence"]
    uncertain = confidence < UNCERTAIN_THRESHOLD

    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "prediction": results[0]["class"],
        "confidence": confidence,
        "uncertain": uncertain,
        "width": image.width,
        "height": image.height,
        "brightness": round(ImageStat.Stat(image.convert("L")).mean[0], 1),
        "inference_ms": ms,
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("prediction=%s confidence=%.3f uncertain=%s time_ms=%s",
             record["prediction"], confidence, uncertain, ms)
    return {"prediction": results[0], "uncertain": uncertain, "top3": results, "inference_ms": ms}