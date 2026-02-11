import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras

MODEL_A = "../models/ocr_ctc_infer_safe_v11.keras"
MODEL_B = "../ocr_ctc_infer_safe_v12.keras"
DATASET_DIR = "../../dataset/test_samples"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12
BATCH = 64

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

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

def preprocess(img):
    gray = cv2.resize(img,(IMG_W,IMG_H))
    gray = cv2.equalizeHist(gray)

    gray = cv2.copyMakeBorder(
        gray,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return gray.astype("float32")/255.0

def load_dataset(dirpath):
    xs,ys=[],[]

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label = os.path.splitext(f)[0]
        path  = os.path.join(dirpath,f)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        xs.append(preprocess(img)[...,None])
        ys.append(label)

    return np.array(xs,np.float32), ys

def ctc_greedy(pred):

    best  = np.argmax(pred,axis=-1)
    blank = pred.shape[-1]-1
    out   = []

    for seq in best:
        collapsed=[]
        prev=-1

        for s in seq:
            if s != prev:
                collapsed.append(s)
                prev = s

        collapsed = [s for s in collapsed if s != blank]

        if not collapsed:
            out.append("")
            continue

        txt = tf.strings.reduce_join(
            num_to_char(np.array(collapsed)+1)
        ).numpy().decode()

        out.append(txt)

    return out

def edit_distance(a,b):
    dp=np.zeros((len(a)+1,len(b)+1),dtype=int)

    for i in range(len(a)+1): dp[i][0]=i
    for j in range(len(b)+1): dp[0][j]=j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j]=min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+(a[i-1]!=b[j-1])
            )

    return dp[-1][-1]

def score(y,dec):

    exact = sum(a==b for a,b in zip(y,dec))

    char_ok = 0
    char_total = 0
    ed = 0

    for gt,pr in zip(y,dec):
        m = min(len(gt),len(pr))
        char_ok += sum(gt[i]==pr[i] for i in range(m))
        char_total += len(gt)
        ed += edit_distance(gt,pr)

    print("Exact accuracy:", round(exact/len(y),4))
    print("Char accuracy :", round(char_ok/char_total,4))
    print("Mean edit dist:", round(ed/len(y),4))

print("Loading dataset...")
X,Y = load_dataset(DATASET_DIR)
print("Images:", len(X))


def eval_model(path):

    print("\n====================")
    print("Model:", path)

    model = keras.models.load_model(
        path,
        compile=False,
        custom_objects={"collapse_hw": collapse_hw}
    )

    logits = model.predict(X,batch_size=BATCH,verbose=1)
    dec = ctc_greedy(logits)

    score(Y,dec)

    print("\nSamples:")
    for i in range(10):
        print(Y[i], "|", dec[i])

    return dec


dec_a = eval_model(MODEL_A)
dec_b = eval_model(MODEL_B)

print("\n====================")
print("Side-by-side samples:")
for i in range(15):
    print(Y[i], "|", dec_a[i], "|", dec_b[i])
