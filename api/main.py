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
MODELS = BASE / "models"
MODEL_PATH = MODELS / "cnn_final.keras"
CLASSES_PATH = MODELS / "classes.json"
REF_PATH = MODELS / "ref_embeddings.npy"
OOD_PATH = MODELS / "ood.json"

IMG = 128                       # يجب أن يطابق حجم الصورة في التدريب
MAX_BYTES = 8 * 1024 * 1024     # أقصى حجم لصورة مرفوعة: 8MB
LOW_CONFIDENCE = 0.60

for p in (MODEL_PATH, CLASSES_PATH, REF_PATH, OOD_PATH):
    if not p.exists():
        raise RuntimeError(f"الملف ناقص: {p.relative_to(BASE)}")

cnn = keras.models.load_model(MODEL_PATH)
# نموذج واحد يعطي: احتمالات الفئات + بصمة الصورة (طبقة embedding) في تمريرة واحدة
multi = keras.Model(cnn.input, [cnn.output, cnn.get_layer("embedding").output])

classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
ref = np.load(REF_PATH).astype("float32")           # (10000, 256)
ood = json.loads(OOD_PATH.read_text(encoding="utf-8"))
THR = float(ood["thr_sim"])
K = int(ood["k"])

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
    arr = np.asarray(im, dtype=np.float32)               # قيم 0-255 (الـ Rescaling داخل النموذج)
    arr = tf.image.resize(arr, (IMG, IMG)).numpy()       # نفس طريقة التدريب
    return arr[None]                                     # (1, IMG, IMG, 3)


def knn_score(f: np.ndarray) -> float:
    """متوسط أعلى K تشابهات cosine مع صور المرجع (نفس دالة Colab)."""
    sim = f @ ref.T
    return float(np.partition(sim, -K, axis=1)[:, -K:].mean(axis=1)[0])


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(classes), "ood_threshold": THR}


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
        probs, emb = multi(x, training=False)
    probs = probs.numpy()[0]
    emb = emb.numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)   # تطبيع L2
    score = knn_score(emb)

    # صورة بعيدة عن صور التدريب: نرفضها بدل تخمين تشخيص
    if score < THR:
        return {"unknown": True, "score": round(score, 3), "threshold": round(THR, 3)}

    top = np.argsort(probs)[::-1][:3]
    results = [{**pretty(classes[i]), "confidence": float(probs[i])} for i in top]
    best = results[0]
    return {
        "unknown": False,
        "score": round(score, 3),
        "prediction": best,
        "top3": results,
        "low_confidence": best["confidence"] < LOW_CONFIDENCE,
    }
