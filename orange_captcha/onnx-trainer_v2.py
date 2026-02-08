import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

# -------------------------
# CONFIG
# -------------------------

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12   # helps CTC emit last characters like "aa"

BATCH = 32
EPOCHS = 40
STEPS_PER_EPOCH = 800

DIR_V2   = "../dataset/generated_samples_v2"
DIR_V3   = "../dataset/generated_samples_v3"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives"

P_V2   = 0.30
P_V3   = 0.30
P_REAL = 0.25
P_HARD = 0.15

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

# -------------------------
# CHAR MAP
# -------------------------

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

# -------------------------
# LOAD DATA
# -------------------------

def load_set(dirpath):
    paths, labels = [], []
    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath, f))
            labels.append(os.path.splitext(f)[0])

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    lens = np.array([len(t) for t in labels], np.int32)

    y = char_to_num(
        tf.strings.unicode_split(labels, "UTF-8")
    ).to_tensor(default_value=0)

    y = tf.maximum(y - 1, 0).numpy().astype(np.int32)

    return imgs, y, lens

v2_imgs, v2_y, v2_l = load_set(DIR_V2)
v3_imgs, v3_y, v3_l = load_set(DIR_V3)
real_imgs, real_y, real_l = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l = load_set(DIR_HARD)

print("Loaded:", len(v2_imgs), len(v3_imgs), len(real_imgs), len(hard_imgs))

# -------------------------
# PREPROCESS
# -------------------------

def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)

    # right padding to help CTC tails
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, RIGHT_PAD,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return img.astype("float32") / 255.0

# -------------------------
# AUGMENT (tail-safe)
# -------------------------

def augment(img, strength):
    img = img.copy()

    if np.random.rand() < 0.5:
        return preprocess(img)

    if np.random.rand() < 0.25 * strength:
        h, w = img.shape
        x = np.random.randint(0, w - 15)
        cv2.rectangle(img, (x, 0), (x + 10, h), 0, -1)

    if np.random.rand() < 0.25 * strength:
        h, w = img.shape
        y = np.random.randint(0, h)
        cv2.line(img, (0, y), (w, y), 0, 10)

    return preprocess(img)

# -------------------------
# SAMPLING
# -------------------------

def sample_source():
    r = np.random.rand()

    if r < P_V2:
        i = np.random.randint(len(v2_imgs))
        return v2_imgs[i], v2_y[i], v2_l[i]

    if r < P_V2 + P_V3:
        i = np.random.randint(len(v3_imgs))
        return v3_imgs[i], v3_y[i], v3_l[i]

    if r < P_V2 + P_V3 + P_REAL:
        i = np.random.randint(len(real_imgs))
        return real_imgs[i], real_y[i], real_l[i]

    i = np.random.randint(len(hard_imgs))
    return hard_imgs[i], hard_y[i], hard_l[i]

epoch_var = tf.Variable(0)

def gen_train():
    while True:
        img, y, l = sample_source()

        e = int(epoch_var.numpy())
        strength = 0.0 if e < 5 else min(1.0, (e-5)/20)

        yield augment(img, strength)[..., None], y, l

train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H, IMG_W+RIGHT_PAD, 1), tf.float32),
        tf.TensorSpec((None,), tf.int32),
        tf.TensorSpec((), tf.int32)
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# -------------------------
# MODEL — ONNX SAFE
# -------------------------

inp = layers.Input(shape=(IMG_H, IMG_W+RIGHT_PAD, 1))

x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
# ❌ removed last pooling to increase time steps

x = layers.Permute((2,1,3))(x)

time_steps = (IMG_W+RIGHT_PAD)//4
feat_dim = (IMG_H//4) * 128

x = layers.Reshape((time_steps, feat_dim))(x)
x = layers.Dense(128, activation="relu")(x)

# ⭐ ONNX SAFE LSTMs
x = layers.Bidirectional(layers.LSTM(
    128,
    return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    unroll=False,
    implementation=2
))(x)

x = layers.Bidirectional(layers.LSTM(
    64,
    return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    unroll=False,
    implementation=2
))(x)

out = layers.Dense(len(characters)+1, activation="softmax")(x)

infer_model = models.Model(inp, out)

# -------------------------
# CTC LOSS
# -------------------------

@tf.function
def ctc_loss(y_true, y_pred, label_len):
    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t * tf.ones((b,1), tf.int32)
    label_len = label_len[:,None]
    return tf.reduce_mean(
        tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, inp_len, label_len
        )
    )

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        x,y,l = data
        with tf.GradientTape() as tape:
            p = self(x, training=True)
            loss = ctc_loss(y,p,l)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))

        return {"loss": loss}

train_model = CTCModel(infer_model.inputs, infer_model.outputs)

train_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4)
)

# -------------------------
# BEAM DECODE (fixes aa loss)
# -------------------------

def decode_beam(pred):
    inp_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded,_ = tf.keras.backend.ctc_decode(
        pred,
        input_length=inp_len,
        greedy=False,
        beam_width=10
    )
    seqs = decoded[0].numpy()

    out=[]
    for s in seqs:
        s = s[s!=-1]
        out.append("".join(num_to_char(s+1).numpy().astype(str)))
    return out

# -------------------------
# TRAIN
# -------------------------

for e in range(EPOCHS):
    epoch_var.assign(e)

    print("\nEpoch", e+1, "/", EPOCHS)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1
    )

# -------------------------
# EXPORT ONNX ONLY
# -------------------------

import tf2onnx

spec = (tf.TensorSpec(
    (None, IMG_H, IMG_W+RIGHT_PAD, 1),
    tf.float32,
    name="image"
),)

tf2onnx.convert.from_keras(
    infer_model,
    input_signature=spec,
    opset=13,
    output_path="ocr_safe.onnx"
)

print("\n✅ Saved ONNX → ocr_safe.onnx")
