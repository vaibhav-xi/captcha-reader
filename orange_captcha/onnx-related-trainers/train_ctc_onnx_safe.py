import os

import cv2
import numpy as np
import string
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import keras
from keras import layers, models

V3_MODEL = "captcha_ctc_adapted_v3.keras"
DATASET_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/dataset_5k"

IMG_W = 200
IMG_H = 50

BATCH_SIZE = 32
EPOCHS = 3
STEPS = 300

characters = string.ascii_letters + string.digits + "@=#"
num_classes = len(characters) + 1

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

@tf.function
def ctc_loss_fn(y_true,y_pred,label_len):
    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t*tf.ones((b,1),tf.int32)
    label_len = label_len[:,None]
    return tf.keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len)
    
class CTCModel(tf.keras.Model):
    def train_step(self, data):
        x,y,l = data
        with tf.GradientTape() as tape:
            p = self(x,training=True)
            loss = ctc_loss_fn(y,p,l)
        g = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))
        return {"loss":tf.reduce_mean(loss)}

def load_data(dirpath):
    paths, labels = [], []
    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath, f))
            labels.append(os.path.splitext(f)[0])

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

    y = char_to_num(
        tf.strings.unicode_split(labels, "UTF-8")
    ).to_tensor(default_value=0)

    y = tf.maximum(y - 1, 0).numpy().astype(np.int32)
    lens = np.array([len(x) for x in labels], np.int32)

    return imgs, y, lens

imgs, ys, lens = load_data(DATASET_DIR)
print("Loaded samples:", len(imgs))

def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    return img.astype("float32") / 255.0

def gen():
    while True:
        i = np.random.randint(len(imgs))
        yield preprocess(imgs[i])[...,None], ys[i], lens[i]

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_cpu_safe_model():
    inp = layers.Input(shape=(IMG_H,IMG_W,1), name="image")

    x = layers.Conv2D(32,3,padding="same",activation="relu")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128,3,padding="same",activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,1))(x)

    x = layers.Permute((2,1,3))(x)
    x = layers.Reshape((50,768))(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=True
        )
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=True
        )
    )(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out)

model = build_cpu_safe_model()

print("Loading v3 model...")
v3 = tf.keras.models.load_model(
    V3_MODEL,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("Copying weights...")
model.set_weights(v3.get_weights())

@tf.function
def ctc_loss(y_true,y_pred,label_len):
    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t*tf.ones((b,1),tf.int32)
    label_len = label_len[:,None]
    return keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len)

ctc_model = CTCModel(model.inputs, model.outputs)
ctc_model.compile(optimizer=keras.optimizers.Adam(1e-5))

print("Fine tuning...")
ctc_model.fit(ds, epochs=EPOCHS, steps_per_epoch=STEPS)

print("Exporting ONNX...")

spec = (tf.TensorSpec((None,IMG_H,IMG_W,1), tf.float32, name="image"),)

onnx_model,_ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17
)

open("captcha_ctc_v3_safe.onnx","wb").write(
    onnx_model.SerializeToString()
)

print("ONNX written")

print("Validating ONNX runtime load...")
ort.InferenceSession("captcha_ctc_v3_safe.onnx")

print("SUCCESS — SAFE ONNX READY")
