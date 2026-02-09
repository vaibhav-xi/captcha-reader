import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
import onnxruntime as ort

KERAS_MODEL = "ocr_ctc_infer_safe.keras"
ONNX_MODEL  = "feature_model.onnx"
DATASET_DIR = "../../dataset/orange-samples"

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
    img = cv2.copyMakeBorder(img,0,0,0,RIGHT_PAD,
                             cv2.BORDER_CONSTANT,value=255)
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
        txt=tf.strings.reduce_join(
            num_to_char(np.array(collapsed)+1)
        ).numpy().decode()
        out.append(txt)
    return out

def score(y,dec):
    exact=sum(a==b for a,b in zip(y,dec))
    print("Exact:", exact/len(y))

print("Loading data...")
X,Y = load_dataset(DATASET_DIR)

print("Loading full Keras model...")
full = keras.models.load_model(
    KERAS_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

feature_model = keras.Model(
    full.input,
    full.layers[-2].output
)

dense = full.layers[-1]
W,b = dense.get_weights()

print("Feature dim:", W.shape[0], "Classes:", W.shape[1])

print("\nKeras baseline...")
k_logits = full.predict(X,batch_size=64,verbose=1)
k_dec = ctc_greedy(k_logits)
score(Y,k_dec)

print("\nRunning ONNX features...")
sess = ort.InferenceSession(ONNX_MODEL)
inp = sess.get_inputs()[0].name

feat=[]
for i in range(0,len(X),64):
    feat.append(sess.run(None,{inp:X[i:i+64]})[0])

feat = np.concatenate(feat,axis=0)

print("Applying Dense head manually...")
o_logits = feat @ W + b 

o_dec = ctc_greedy(o_logits)

print("\nONNX+Dense result...")
score(Y,o_dec)

print("\nSamples:")
for i in range(10):
    print(Y[i], "|", o_dec[i])
