import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
import onnxruntime as ort

KERAS_MODEL = "ocr_ctc_infer.keras"
ONNX_MODEL  = "model2.onnx"
DATASET_DIR = "../dataset/orange-samples"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = keras.layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

def preprocess(img):
    img = cv2.resize(img,(IMG_W,IMG_H))
    img = cv2.equalizeHist(img)

    img = cv2.copyMakeBorder(
        img, 0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return img.astype("float32")/255.0

def load_dataset(dirpath):
    xs = []
    ys = []

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label = os.path.splitext(f)[0]
        path  = os.path.join(dirpath, f)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        xs.append(preprocess(img)[...,None])
        ys.append(label)

    return np.array(xs, np.float32), ys

def ctc_beam_decode(pred):
    L = np.ones(pred.shape[0]) * pred.shape[1]

    decoded,_ = tf.keras.backend.ctc_decode(
        pred,
        L,
        greedy=False,
        beam_width=12,
        top_paths=1
    )

    decoded = decoded[0].numpy()

    out = []
    for seq in decoded:
        seq = seq[seq != -1]
        txt = tf.strings.reduce_join(num_to_char(seq+1)).numpy().decode()
        out.append(txt)

    return out

def edit_distance(a,b):
    dp = np.zeros((len(a)+1,len(b)+1),dtype=int)

    for i in range(len(a)+1):
        dp[i][0]=i
    for j in range(len(b)+1):
        dp[0][j]=j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j] = min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1] + (a[i-1]!=b[j-1])
            )

    return dp[-1][-1]

def eval_keras(x, y):
    print("\n====================")
    print("KERAS MODEL EVAL")
    print("====================")

    model = keras.models.load_model(KERAS_MODEL)

    pred = model.predict(x, batch_size=64, verbose=1)
    dec  = ctc_beam_decode(pred)

    exact = 0
    char_ok = 0
    char_total = 0
    ed_sum = 0

    for gt,pr in zip(y,dec):
        if gt == pr:
            exact += 1

        m = min(len(gt),len(pr))
        char_ok += sum(gt[i]==pr[i] for i in range(m))
        char_total += len(gt)

        ed_sum += edit_distance(gt,pr)

    print("Exact accuracy:", exact/len(y))
    print("Char accuracy :", char_ok/char_total)
    print("Mean edit dist:", ed_sum/len(y))

    print("\nSamples:")
    for i in range(10):
        print("GT:",y[i],"| PR:",dec[i])

def eval_onnx(x, y):
    print("\n====================")
    print("ONNX MODEL EVAL")
    print("====================")

    sess = ort.InferenceSession(
        ONNX_MODEL,
        providers=["CPUExecutionProvider"]
    )

    inp_name = sess.get_inputs()[0].name

    preds = []

    for i in range(0,len(x),64):
        batch = x[i:i+64]
        out = sess.run(None, {inp_name: batch})[0]
        preds.append(out)

    pred = np.concatenate(preds, axis=0)
    dec  = ctc_beam_decode(pred)

    exact = 0
    char_ok = 0
    char_total = 0
    ed_sum = 0

    for gt,pr in zip(y,dec):
        if gt == pr:
            exact += 1

        m = min(len(gt),len(pr))
        char_ok += sum(gt[i]==pr[i] for i in range(m))
        char_total += len(gt)

        ed_sum += edit_distance(gt,pr)

    print("Exact accuracy:", exact/len(y))
    print("Char accuracy :", char_ok/char_total)
    print("Mean edit dist:", ed_sum/len(y))

    print("\nSamples:")
    for i in range(10):
        print("GT:",y[i],"| PR:",dec[i])


if __name__ == "__main__":

    print("Loading dataset...")
    X,Y = load_dataset(DATASET_DIR)
    print("Images:", len(X))

    eval_keras(X,Y)
    eval_onnx(X,Y)
