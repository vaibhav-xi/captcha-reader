import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import models
import shutil   

DATASET_DIR = "../dataset/generated_samples_v3"
MODEL_PATH  = "ocr_ctc_infer_safe.keras"
HARD_DIR    = "../dataset/hard_negatives_v3"   

IMG_W = 200
IMG_H = 50

RIGHT_PAD = 12 

TTA_RUNS = 12
TTA_RUNS_HARD = 24
CONF_THRESHOLD = 0.82

characters = string.ascii_letters + string.digits + "@=#"
idx_to_char = {i: c for i, c in enumerate(characters)}
blank_index = len(characters)

os.makedirs(HARD_DIR, exist_ok=True)   

class CTCModel(tf.keras.Model):
    pass

@tf.keras.utils.register_keras_serializable()
def collapse_hw(t):
    s = tf.shape(t)
    return tf.reshape(t, [s[0], s[1], s[2] * s[3]])

model = models.load_model(
    MODEL_PATH,
    custom_objects={
        "CTCModel": CTCModel,
        "collapse_hw": collapse_hw
    },
    compile=False
)

print("Model output shape:", model.output_shape)

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)

    img = cv2.copyMakeBorder(
        img, 0, 0, 0, RIGHT_PAD,
        cv2.BORDER_CONSTANT,
        value=255
    )

    img = img.astype("float32") / 255.0
    return img

def tta_variant(img):
    v = img.copy()

    if np.random.rand() < 0.5:
        v = cv2.dilate(v, np.ones((2,2),np.uint8))
    else:
        v = cv2.erode(v, np.ones((2,2),np.uint8))

    alpha = np.random.uniform(0.9, 1.15)
    beta  = np.random.uniform(-0.05, 0.05)
    v = np.clip(v * alpha + beta, 0, 1)

    if np.random.rand() < 0.5:
        c = np.random.randint(2,10)
        v = v[:, c:-c]
        v = cv2.resize(v, (IMG_W + RIGHT_PAD, IMG_H))

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
    return best, final_conf

def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j] = min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1] + (a[i-1]!=b[j-1])
            )
    return dp[-1][-1]

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))

correct = 0
cer_total = 0
len_correct = 0
hard_saved = 0   

print("\nRunning TTA inference...")

for path, truth in zip(paths, labels):

    img = preprocess(path)

    pred, conf = tta_predict(img, TTA_RUNS)
    if conf < CONF_THRESHOLD:
        pred, conf = tta_predict(img, TTA_RUNS_HARD)

    # -------------------------------
    # HARD NEGATIVE SAVE  
    # -------------------------------
    if pred != truth:
        dst = os.path.join(HARD_DIR, os.path.basename(path))
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
            hard_saved += 1
    # -------------------------------

    if pred == truth:
        correct += 1

    if len(pred) == len(truth):
        len_correct += 1

    cer_total += edit_distance(truth, pred) / len(truth)

    print(f"{truth:10s} → {pred:10s}  conf={conf:.2f}")

acc = correct / len(labels)
cer = cer_total / len(labels)
len_acc = len_correct / len(labels)

print("\n====================")
print("FULL CAPTCHA ACC:", round(acc,4))
print("CHAR ERROR RATE :", round(cer,4))
print("LENGTH ACCURACY :", round(len_acc,4))
print("HARD NEGATIVES SAVED:", hard_saved)   
print("====================")
