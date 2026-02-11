import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, Model

DATASET_DIR = "../dataset/generated_samples_v3"

IMG_W = 200
IMG_H = 50

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

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img

X = np.array([load_img(p) for p in paths])
X = X[..., np.newaxis]

label_lengths = np.array([len(t) for t in labels])

y = char_to_num(
    tf.strings.unicode_split(labels, "UTF-8")
).to_tensor(default_value=0)

y = tf.maximum(y - 1, 0).numpy()

idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(len(X) * 0.8)

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
len_train, len_test = label_lengths[idx[:split]], label_lengths[idx[split:]]

inp = layers.Input(shape=(IMG_H, IMG_W, 1), name="image")

x = layers.Conv2D(32,3,activation="relu",padding="same")(inp)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D((2,1))(x)

new_shape = (IMG_W // 4, (IMG_H // 4) * 128)
x = layers.Permute((2,1,3))(x)
x = layers.Reshape((50, 6*128))(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

out = layers.Dense(len(characters)+1, activation="softmax")(x)

model = Model(inp, out)

def make_ctc_loss(label_lens):

    label_lens = tf.constant(label_lens, dtype=tf.int32)

    def loss(y_true, y_pred):

        batch = tf.shape(y_pred)[0]
        input_len = tf.shape(y_pred)[1]

        input_len = input_len * tf.ones((batch,1), dtype=tf.int32)
        label_len = label_lens[:batch][:,None]

        return tf.keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_len,
            label_len
        )

    return loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=make_ctc_loss(len_train)
)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=60
)

model.save("captcha_ctc_model_60.keras")
