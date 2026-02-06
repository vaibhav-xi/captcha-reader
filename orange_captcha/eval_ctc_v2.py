import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import models

DATASET_DIR = "../dataset/generated_samples_v3"
MODEL_PATH = "captcha_ctc_model_v2.keras"

IMG_W = 200
IMG_H = 50

characters = string.ascii_letters + string.digits + "@=#"

idx_to_char = {i: c for i, c in enumerate(characters)}
blank_index = len(characters)

class CTCModel(tf.keras.Model):

    def train_step(self, data):
        x, y, label_len = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.reduce_mean(y_pred)
        return {"loss": loss}

    def test_step(self, data):
        return {"loss": 0.0}
    
# model = models.load_model(MODEL_PATH, compile=False)

model = models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("\nModel output shape:", model.output_shape)

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))

X = np.array([load_img(p) for p in paths])
X = X[..., np.newaxis]

def decode_batch(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded, _ = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )

    decoded = decoded[0].numpy()

    texts = []

    for seq in decoded:
        chars = []

        for idx in seq:

            if idx == -1:
                continue

            if idx == blank_index:
                continue

            if idx in idx_to_char:
                chars.append(idx_to_char[idx])

        texts.append("".join(chars))

    return texts

def edit_distance(a, b):

    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1):
        dp[i][0] = i

    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (a[i-1] != b[j-1])
            )

    return dp[-1][-1]

print("\nRunning inference...")
pred = model.predict(X, batch_size=64, verbose=1)

decoded = decode_batch(pred)

correct = 0
cer_total = 0
len_correct = 0

for truth, pred_txt in zip(labels, decoded):

    if pred_txt == truth:
        correct += 1

    if len(pred_txt) == len(truth):
        len_correct += 1

    cer_total += edit_distance(truth, pred_txt) / len(truth)

acc = correct / len(labels)
cer = cer_total / len(labels)
len_acc = len_correct / len(labels)

print("\n====================")
print("FULL CAPTCHA ACC:", round(acc, 4))
print("CHAR ERROR RATE :", round(cer, 4))
print("LENGTH ACCURACY :", round(len_acc, 4))
print("====================\n")

errors = []

for truth, pred_txt, path in zip(labels, decoded, paths):
    if truth != pred_txt:
        errors.append((truth, pred_txt, path))

print("Sample errors:\n")

for t, p, path in errors[:12]:
    print("GT :", t)
    print("PR :", p)
    print(path)
    print("---")
