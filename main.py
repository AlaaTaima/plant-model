import io
import json
import os
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "cnn_best.keras"
CLASSES_PATH = BASE / "models" / "classes.json"

IMG = 128                       # يجب أن يطابق حجم الصورة في التدريب
MAX_BYTES = 8 * 1024 * 1024     # أقصى حجم لصورة مرفوعة: 8MB
LOW_CONFIDENCE = 0.60

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    raise RuntimeError(
        "Missing models/cnn_best.keras or models/classes.json. "
        "Export them from Colab and put them in the models/ folder."
    )

model = keras.models.load_model(MODEL_PATH)
classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
lock = threading.Lock()

app = FastAPI(title="Plant Disease Classifier")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def pretty(label: str) -> dict:
    """'Corn_(maize)___Northern_Leaf_Blight' -> plant + condition."""
    plant, _, cond = label.partition("___")
    clean = lambda s: " ".join(s.replace("_", " ").split())
    return {
        "label": label,
        "plant": clean(plant),
        "condition": clean(cond),
        "healthy": cond.strip().lower() == "healthy",
    }


def preprocess(raw: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="الملف ليس صورة صالحة")
    arr = np.asarray(im, dtype=np.float32)               # قيم 0-255
    arr = tf.image.resize(arr, (IMG, IMG)).numpy()       # نفس طريقة التدريب
    return arr[None]                                     # (1, IMG, IMG, 3)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(classes)}


# دالة عادية (غير async) كي تعمل في thread ولا تحجب السيرفر
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="الصورة كبيرة (الحد 8MB)")
    if not raw:
        raise HTTPException(status_code=400, detail="الملف فارغ")

    x = preprocess(raw)
    with lock:
        probs = model(x, training=False).numpy()[0]

    top = np.argsort(probs)[::-1][:3]
    results = [{**pretty(classes[i]), "confidence": float(probs[i])} for i in top]
    best = results[0]
    return {
        "prediction": best,
        "top3": results,
        "low_confidence": best["confidence"] < LOW_CONFIDENCE,
    }
