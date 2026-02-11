import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
from collections import Counter, defaultdict
import csv

MODEL_PATH  = "ocr_ctc_infer_safe_v11.keras"
DATASET_DIR = "../dataset/test_samples"

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

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

def preprocess(img):
    img = cv2.resize(img,(IMG_W,IMG_H))
    img = cv2.equalizeHist(img)
    img = cv2.copyMakeBorder(
        img,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )
    return img.astype("float32")/255.0

def load_dataset(dirpath):
    xs,ys=[],[]

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label=os.path.splitext(f)[0]
        img=cv2.imread(
            os.path.join(dirpath,f),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        xs.append(preprocess(img)[...,None])
        ys.append(label)

    return np.array(xs,np.float32), ys

def ctc_decode_batch(pred):

    best = np.argmax(pred,axis=-1)
    blank = pred.shape[-1]-1

    out=[]

    for seq in best:

        collapsed=[]
        prev=-1

        for s in seq:
            if s!=prev:
                collapsed.append(s)
                prev=s

        collapsed=[s for s in collapsed if s!=blank]

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

    for i in range(len(a)+1):
        dp[i][0]=i
    for j in range(len(b)+1):
        dp[0][j]=j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j]=min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+(a[i-1]!=b[j-1])
            )

    return dp[-1][-1]

def align_pairs(a,b):
    """simple alignment for confusion counting"""
    m=min(len(a),len(b))
    return list(zip(a[:m], b[:m]))

def analyze(gt_list, pr_list):

    exact = 0
    char_ok = 0
    char_total = 0
    ed_total = 0

    confusion = Counter()
    len_errors = 0
    double_letter_errors = 0

    thin_set = set("Il1tf")

    thin_miss = 0
    thin_total = 0

    for gt,pr in zip(gt_list, pr_list):

        if gt == pr:
            exact += 1

        if len(gt) != len(pr):
            len_errors += 1

        ed_total += edit_distance(gt,pr)

        m = min(len(gt), len(pr))
        char_total += len(gt)

        for i in range(m):

            if gt[i] == pr[i]:
                char_ok += 1
            else:
                confusion[(gt[i],pr[i])] += 1

        for c in gt:
            if c in thin_set:
                thin_total += 1

        for g,p in align_pairs(gt,pr):
            if g in thin_set and g != p:
                thin_miss += 1

        for a,b in zip(gt, gt[1:]):
            if a == b:
                if pr.find(a+a) == -1:
                    double_letter_errors += 1
                    break

    print("\n====================")
    print("FULL CAPTCHA ACC:", exact/len(gt_list))
    print("CHAR ACC:", char_ok/char_total)
    print("MEAN EDIT DIST:", ed_total/len(gt_list))
    print("LEN ERR RATE:", len_errors/len(gt_list))

    if thin_total > 0:
        print("THIN-STROKE MISS RATE:", thin_miss/thin_total)

    print("DOUBLE LETTER FAILS:", double_letter_errors)
    print("====================")

    return confusion

print("Loading dataset...")
X,Y = load_dataset(DATASET_DIR)
print("Images:", len(X))

print("\nLoading model...")
model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"collapse_hw": collapse_hw})

print("Running inference...")
logits = model.predict(X, batch_size=64, verbose=1)

decoded = ctc_decode_batch(logits)

conf = analyze(Y, decoded)

print("\nTop confusions:")
for (g,p),n in conf.most_common(20):
    print(f"{g} -> {p} : {n}")
    
with open("confusions.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["gt","pred","count"])
    for (g,p),n in conf.items():
        w.writerow([g,p,n])

print("\nSaved confusions.csv")
