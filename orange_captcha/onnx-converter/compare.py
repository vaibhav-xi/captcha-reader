import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
import onnxruntime as ort

KERAS_MODEL = "ocr_ctc_onnx_safe.keras"
ONNX_MODEL  = "captcha.onnx"
DATASET_DIR = "../../dataset/new_500"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12
BATCH = 32

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
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g,(IMG_W,IMG_H))
    g = cv2.equalizeHist(g)
    g = cv2.copyMakeBorder(
        g,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )
    return g.astype("float32")/255.0

def load_dataset(dirpath, limit=128):
    xs, ys = [], []

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(dirpath,f))
        if img is None:
            continue
        xs.append(preprocess(img)[...,None])
        ys.append(os.path.splitext(f)[0])
        if len(xs) >= limit:
            break

    return np.array(xs,np.float32), ys

def ctc_greedy(pred):
    best = pred.argmax(-1)
    blank = pred.shape[-1]-1
    out = []

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
    import numpy as np
    dp = np.zeros((len(a)+1,len(b)+1), int)
    dp[:,0] = np.arange(len(a)+1)
    dp[0,:] = np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i,j] = min(
                dp[i-1,j]+1,
                dp[i,j-1]+1,
                dp[i-1,j-1] + (a[i-1]!=b[j-1])
            )
    return dp[-1,-1]

print("Loading dataset...")
X,Y = load_dataset(DATASET_DIR)
print("Samples:", len(X))

print("\nLoading Keras...")
kmodel = keras.models.load_model(
    KERAS_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

k_logits = kmodel.predict(X, batch_size=BATCH, verbose=0)

print("Keras output shape:", k_logits.shape)

print("\nLoading ONNX...")
sess = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

print("ONNX input :", inp_name, sess.get_inputs()[0].shape)
print("ONNX output:", out_name, sess.get_outputs()[0].shape)

o_logits = sess.run(
    [out_name],
    {inp_name: X}
)[0]

print("ONNX output shape:", o_logits.shape)

diff = np.abs(k_logits - o_logits)

print("\nNumeric drift:")
print("max :", diff.max())
print("mean:", diff.mean())

k_arg = k_logits.argmax(-1)
o_arg = o_logits.argmax(-1)

arg_mismatch = np.mean(k_arg != o_arg)

print("\nArgmax mismatch rate:", arg_mismatch)

k_txt = ctc_greedy(k_logits)
o_txt = ctc_greedy(o_logits)

txt_match = np.mean([a==b for a,b in zip(k_txt,o_txt)])

print("Decoded text match:", txt_match)

cer = np.mean([
    edit_distance(a,b)/max(1,len(a))
    for a,b in zip(k_txt,o_txt)
])

print("Char error rate:", cer)

print("\nSamples:")
for i in range(min(15, len(Y))):
    print(Y[i], "|", k_txt[i], "|", o_txt[i])
