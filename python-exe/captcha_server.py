import os
import base64
import uuid
import platform
from pathlib import Path

import cv2
import numpy as np
import string
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras import models
import uvicorn
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "captcha_ctc_adapted_v3.keras"

IMG_W = 200
IMG_H = 50

TTA_RUNS = 6
TTA_RUNS_HARD = 12
CONF_THRESHOLD = 0.82

ENV_SAVE_PATH = "CAPTCHA_SAVE_PATH"

def get_default_save_dir():
    custom = os.getenv(ENV_SAVE_PATH)
    if custom:
        return Path(custom)

    user = Path.home()

    if platform.system() == "Windows":
        return user / "Documents" / "captchas"
    else:
        return user / "captchas"


SAVE_DIR = get_default_save_dir()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

characters = string.ascii_letters + string.digits + "@=#"
idx_to_char = {i: c for i, c in enumerate(characters)}
blank_index = len(characters)

class CTCModel(tf.keras.Model):
    pass

print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

model = models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

_dummy = np.zeros((1, IMG_H, IMG_W, 1), dtype=np.float32)
model.predict(_dummy, verbose=0)

app = FastAPI(title="Captcha OCR API")

class Base64Request(BaseModel):
    base64: str
    file_name: str | None = None

def save_base64_png(b64_string: str, file_name: str | None):
    comma = b64_string.find(",")
    if comma > 0:
        b64_string = b64_string[comma+1:]

    try:
        data = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(400, "Invalid base64")

    name = file_name or str(uuid.uuid4())
    if not name.lower().endswith(".png"):
        name += ".png"

    path = SAVE_DIR / name

    with open(path, "wb") as f:
        f.write(data)

    return str(path)


def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image unreadable")

    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img


def tta_variant(img):
    v = img.copy()

    if np.random.rand() < 0.5:
        v = cv2.dilate(v, np.ones((2,2),np.uint8))
    else:
        v = cv2.erode(v, np.ones((2,2),np.uint8))

    alpha = np.random.uniform(0.9, 1.1)
    beta  = np.random.uniform(-0.03, 0.03)

    v = np.clip(v * alpha + beta, 0, 1)
    return v


def decode_with_conf(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded, _ = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )

    seqs = decoded[0].numpy()

    texts = []
    confs = []

    for b, seq in enumerate(seqs):
        chars = []
        probs = []

        for t, idx in enumerate(seq):
            if idx == -1 or idx == blank_index:
                continue

            if idx in idx_to_char:
                chars.append(idx_to_char[idx])
                probs.append(np.max(pred[b, t]))

        conf = float(np.mean(probs)) if probs else 0.0
        texts.append("".join(chars))
        confs.append(conf)

    return texts, confs


def tta_predict(img, runs):
    variants = [tta_variant(img) for _ in range(runs)]
    batch = np.array(variants)[..., None]

    pred = model.predict(batch, verbose=0)
    texts, confs = decode_with_conf(pred)

    best = max(set(texts), key=texts.count)
    vote_conf = texts.count(best) / runs
    mean_conf = np.mean([c for t,c in zip(texts,confs) if t == best])

    final_conf = 0.6*vote_conf + 0.4*mean_conf
    return best, float(final_conf)

@app.post("/predict_base64")
def predict_base64(req: Base64Request):
    try:
        path = save_base64_png(req.base64, req.file_name)

        img = preprocess(path)

        pred, conf = tta_predict(img, TTA_RUNS)

        if conf < CONF_THRESHOLD:
            pred, conf = tta_predict(img, TTA_RUNS_HARD)

        return {
            "prediction": pred,
            "confidence": round(conf, 3),
            "saved_to": path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    print("Save directory:", SAVE_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8000)
