import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

KERAS_MODEL = "ocr_ctc_infer_safe_v9.keras"
FROZEN_PB   = "captcha_frozen.pb"
DATASET_DIR = "../../dataset/new_500"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12
BATCH = 32

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

def load_dataset(dirpath, limit=64):
    xs=[]

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        img = cv2.imread(os.path.join(dirpath,f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        xs.append(preprocess(img)[...,None])

        if len(xs) >= limit:
            break

    return np.array(xs,np.float32)

print("Loading keras...")
kmodel = keras.models.load_model(
    KERAS_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Loading frozen graph...")

with tf.io.gfile.GFile(FROZEN_PB, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

def wrap_frozen_graph(graph_def, inputs, outputs):

    def _imports():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped = tf.compat.v1.wrap_function(_imports, [])
    import_graph = wrapped.graph

    return wrapped.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs),
    )

INPUT_NODE  = "x:0"
OUTPUT_NODE = "Identity:0"

frozen_fn = wrap_frozen_graph(
    graph_def,
    inputs=[INPUT_NODE],
    outputs=[OUTPUT_NODE],
)

X = load_dataset(DATASET_DIR, limit=128)

print("Samples:", len(X))

print("Running keras...")
keras_logits = kmodel.predict(X, batch_size=BATCH)

print("Running frozen...")
frozen_logits = frozen_fn(tf.constant(X))[0].numpy()

print("Shapes:")
print("keras :", keras_logits.shape)
print("frozen:", frozen_logits.shape)

diff = np.abs(keras_logits - frozen_logits)

print("\nNumeric drift:")
print("max :", diff.max())
print("mean:", diff.mean())

k_arg = keras_logits.argmax(-1)
f_arg = frozen_logits.argmax(-1)

mismatch = np.mean(k_arg != f_arg)

print("\nArgmax mismatch rate:", mismatch)
