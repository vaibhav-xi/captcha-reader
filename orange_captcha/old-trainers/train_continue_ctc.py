import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

DATASET_DIR = "../dataset/generated_samples_v3"
MODEL_PATH  = "captcha_ctc_model.keras"

IMG_W = 200
IMG_H = 50

BATCH_SIZE = 32
EPOCHS = 40

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))

# -------------------------

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.copyMakeBorder(img, 0, 0, 6, 6, cv2.BORDER_CONSTANT, value=255)

    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img

X = np.array([load_img(p) for p in paths])[..., np.newaxis]

label_lengths = np.array([len(t) for t in labels], dtype=np.int32)

y = char_to_num(
    tf.strings.unicode_split(labels, "UTF-8")
).to_tensor(default_value=0)

y = tf.maximum(y - 1, 0).numpy().astype(np.int32)

idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(len(X) * 0.9)

X_train, X_val = X[idx[:split]], X[idx[split:]]
y_train, y_val = y[idx[:split]], y[idx[split:]]
len_train, len_val = label_lengths[idx[:split]], label_lengths[idx[split:]]

def make_dataset(X, y, lengths):

    ds = tf.data.Dataset.from_tensor_slices((X, y, lengths))

    ds = ds.shuffle(10000)

    ds = ds.batch(BATCH_SIZE)

    return ds

train_ds = make_dataset(X_train, y_train, len_train)
val_ds   = make_dataset(X_val,   y_val,   len_val)

model = models.load_model(MODEL_PATH, compile=False)

print("\nLoaded model:")
model.summary()

@tf.function
def ctc_loss_fn(y_true, y_pred, label_len):

    batch = tf.shape(y_pred)[0]
    input_len = tf.shape(y_pred)[1]

    input_len = input_len * tf.ones((batch,1), dtype=tf.int32)
    label_len = label_len[:,None]

    return tf.keras.backend.ctc_batch_cost(
        y_true,
        y_pred,
        input_len,
        label_len
    )

class CTCModel(tf.keras.Model):

    def train_step(self, data):
        x, y, label_len = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = ctc_loss_fn(y, y_pred, label_len)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": tf.reduce_mean(loss)}

    def test_step(self, data):
        x, y, label_len = data
        y_pred = self(x, training=False)
        loss = ctc_loss_fn(y, y_pred, label_len)
        return {"loss": tf.reduce_mean(loss)}

ctc_model = CTCModel(model.inputs, model.outputs)

ctc_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4)
)

ctc_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

ctc_model.save("captcha_ctc_model_v2.keras")

print("\nSaved updated model → captcha_ctc_model_v2.keras")
