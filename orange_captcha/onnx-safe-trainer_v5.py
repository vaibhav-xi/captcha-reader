import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# CONFIG
# =========================

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 12              # fine-tune epochs
STEPS_PER_EPOCH = 400    # smaller for fine-tune
VAL_STEPS = 120

DIR_V2   = "../dataset/generated_samples_v2"
DIR_V3   = "../dataset/generated_samples_v3"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives"

# sampling mix
P_V2   = 0.30
P_V3   = 0.30
P_REAL = 0.25
P_HARD = 0.15

# optional — load previous keras weights
PRETRAINED = "ocr_ctc_infer.keras"

# =========================
# CHARSET
# =========================

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

# =========================
# DATA
# =========================

def has_double_char(s):
    return any(a == b for a, b in zip(s, s[1:]))

def load_set(dirpath):
    paths, labels = [], []

    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath, f))
            labels.append(os.path.splitext(f)[0])

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

    y = char_to_num(
        tf.strings.unicode_split(labels, "UTF-8")
    ).to_tensor(default_value=-1)

    y = (y - 1).numpy().astype(np.int32)
    lens = np.array([len(t) for t in labels], np.int32)
    dbl  = np.array([has_double_char(t) for t in labels])

    return imgs, y, lens, dbl


v2_imgs, v2_y, v2_l, v2_d = load_set(DIR_V2)
v3_imgs, v3_y, v3_l, v3_d = load_set(DIR_V3)
real_imgs, real_y, real_l, real_d = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l, hard_d = load_set(DIR_HARD)

print("Loaded sets:",
      len(v2_imgs), len(v3_imgs),
      len(real_imgs), len(hard_imgs))

# =========================
# PREPROCESS / AUGMENT
# =========================

def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)

    img = cv2.copyMakeBorder(
        img, 0, 0, 0, RIGHT_PAD,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return img.astype("float32") / 255.0


def augment(img, strength):
    img = img.copy()

    if np.random.rand() < 0.25:
        return preprocess(img)

    if np.random.rand() < 0.3 * strength:
        h, w = img.shape
        x = np.random.randint(0, w - 12)
        cv2.rectangle(img, (x, 0), (x + 12, h), 0, -1)

    if np.random.rand() < 0.3 * strength:
        h, w = img.shape
        y = np.random.randint(0, h)
        cv2.line(img, (0, y), (w, y), 0, 10)

    return preprocess(img)

# =========================
# SAMPLING
# =========================

def pick(imgs, y, l, dbl):
    if np.random.rand() < 0.35 and np.any(dbl):
        i = np.random.choice(np.where(dbl)[0])
    else:
        i = np.random.randint(len(imgs))
    return imgs[i], y[i], l[i]


def sample_source():
    r = np.random.rand()

    if r < P_V2:
        return pick(v2_imgs, v2_y, v2_l, v2_d)
    if r < P_V2 + P_V3:
        return pick(v3_imgs, v3_y, v3_l, v3_d)
    if r < P_V2 + P_V3 + P_REAL:
        return pick(real_imgs, real_y, real_l, real_d)

    return pick(hard_imgs, hard_y, hard_l, hard_d)

# =========================
# DATASETS
# =========================

epoch_var = tf.Variable(0)

def gen_train():
    while True:
        img, y, l = sample_source()
        strength = min(1.0, float(epoch_var.numpy()) / 20.0)
        yield augment(img, strength)[..., None], y, l


def gen_val():
    while True:
        i = np.random.randint(len(real_imgs))
        yield preprocess(real_imgs[i])[..., None], real_y[i], real_l[i]


train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H, IMG_W + RIGHT_PAD, 1), tf.float32),
        tf.TensorSpec((None,), tf.int32),
        tf.TensorSpec((), tf.int32),
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    gen_val,
    output_signature=(
        tf.TensorSpec((IMG_H, IMG_W + RIGHT_PAD, 1), tf.float32),
        tf.TensorSpec((None,), tf.int32),
        tf.TensorSpec((), tf.int32),
    )
).batch(BATCH)

# debug batch print
for xb, yb, lb in train_ds.take(1):
    print("Batch X:", xb.shape)
    print("Batch Y:", yb.shape)
    print("Batch lens tensor:", lb)

# =========================
# MODEL — ONNX SAFE GRAPH
# =========================

inp = layers.Input(
    shape=(IMG_H, IMG_W + RIGHT_PAD, 1),
    name="image"
)

x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 1))(x)

x = layers.Permute((2, 1, 3))(x)

# ✅ ONNX SAFE dynamic collapse
def collapse_hw(t):
    s = tf.shape(t)
    return tf.reshape(t, [s[0], s[1], s[2] * s[3]])

x = layers.Lambda(collapse_hw, name="collapse_hw")(x)

x = layers.Dense(128, activation="relu")(x)

# CuDNN-free LSTM
x = layers.Bidirectional(layers.LSTM(
    128, return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    unroll=True
))(x)

x = layers.Bidirectional(layers.LSTM(
    64, return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    unroll=True
))(x)

out = layers.Dense(len(characters) + 1, activation="softmax")(x)

infer_model = models.Model(inp, out)

# =========================
# LOAD PRETRAINED (PARTIAL)
# =========================

LEGACY_H5 = "old_model_legacy.h5"

if os.path.exists(LEGACY_H5):
    print("Loading pretrained legacy H5 weights (partial)...")

    infer_model.load_weights(
        LEGACY_H5,
        by_name=True,
        skip_mismatch=True
    )

# freeze conv blocks for fast fine-tune
for layer in infer_model.layers[:6]:
    layer.trainable = False

# =========================
# CTC LOSS
# =========================

def ctc_loss_fn(y_true, y_pred, label_len):
    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t * tf.ones((b, 1), tf.int32)
    label_len = label_len[:, None]
    return tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, inp_len, label_len
    )

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        x, y, l = data
        with tf.GradientTape() as tape:
            p = self(x, training=True)
            loss = tf.reduce_mean(ctc_loss_fn(y, p, l))

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))

        grad_norm = tf.linalg.global_norm(
            [gg for gg in g if gg is not None]
        )

        return {"loss": loss, "grad_norm": grad_norm}

train_model = CTCModel(infer_model.inputs, infer_model.outputs)

train_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=lambda y_true, y_pred: tf.constant(0.0)
)

# =========================
# TRAIN
# =========================

for e in range(EPOCHS):
    epoch_var.assign(e)

    print("\n========================")
    print("Epoch", e + 1, "/", EPOCHS)

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        epochs=1
    )

# =========================
# SAVE + ONNX EXPORT
# =========================

infer_model.save("ocr_ctc_infer_safe.keras")

import tf2onnx

spec = (infer_model.inputs[0],)

tf2onnx.convert.from_keras(
    infer_model,
    input_signature=spec,
    opset=13,
    output_path="ocr_safe.onnx"
)

print("\nSaved → ocr_safe.onnx")
