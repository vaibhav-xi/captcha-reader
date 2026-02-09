import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras

KERAS_MODEL   = "ocr_ctc_infer_safe.keras"
FEATURES_DIR  = "saved_features"
DATASET_DIR   = "../../dataset/orange-samples"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

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
        img,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )
    return img.astype("float32")/255.0

def load_dataset(dirpath):
    xs,ys=[],[]
    for f in os.listdir(dirpath):
        if not f.endswith(".png"): continue
        label=os.path.splitext(f)[0]
        img=cv2.imread(os.path.join(dirpath,f),cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        xs.append(preprocess(img)[...,None])
        ys.append(label)
    return np.array(xs,np.float32), ys

def ctc_greedy(pred):

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
    exact=sum(a==b for a,b in zip(y,dec))
    char_ok=0
    char_total=0
    ed=0

    for gt,pr in zip(y,dec):
        m=min(len(gt),len(pr))
        char_ok += sum(gt[i]==pr[i] for i in range(m))
        char_total += len(gt)
        ed += edit_distance(gt,pr)

    print("Exact:", exact/len(y))
    print("Char :", char_ok/char_total)
    print("Edit :", ed/len(y))

print("Loading dataset...")
X,Y = load_dataset(DATASET_DIR)
print("Images:", len(X))

print("\nLoading full Keras model...")
full = keras.models.load_model(
    KERAS_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

# baseline
print("\nKeras baseline...")
k_logits = full.predict(X,batch_size=64,verbose=1)
k_dec = ctc_greedy(k_logits)
score(Y,k_dec)

dense = full.layers[-1]
W,b = dense.get_weights()
print("\nDense W,b shapes:", W.shape, b.shape)

print("\nLoading saved_features TF model...")
feat_model = tf.saved_model.load(FEATURES_DIR)

infer = feat_model.signatures["serve"]

print("Running feature model...")

feats=[]
for i in range(0,len(X),64):
    batch = tf.constant(X[i:i+64])
    out = infer(batch)
    # get first output tensor
    t = list(out.values())[0].numpy()
    feats.append(t)

feats = np.concatenate(feats,axis=0)

print("Feature tensor shape:", feats.shape)

print("Applying Dense head...")
o_logits = feats @ W + b

print("\nSaved_features pipeline result...")
o_dec = ctc_greedy(o_logits)
score(Y,o_dec)

print("\nSamples:")
for i in range(10):
    print(Y[i], "|", o_dec[i])
