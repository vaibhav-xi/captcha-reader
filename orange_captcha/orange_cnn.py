import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, Model

DATASET_DIR = "../dataset/orange-samples"

IMG_W = 200
IMG_H = 50

characters = string.ascii_letters + string.digits + "@=#"
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True,
)

# LOAD DATA

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        label = os.path.splitext(f)[0]
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(label)

print("Samples:", len(paths))

# IMAGE PREPROCESS

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img


X = np.array([load_img(p) for p in paths])
X = X[..., np.newaxis]

label_lengths = np.array([len(t) for t in labels])
max_label_len = max(label_lengths)

y = char_to_num(tf.strings.unicode_split(labels, "UTF-8")).to_tensor()
y = y.numpy()

# SPLIT

idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(len(X) * 0.8)

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
len_train, len_test = label_lengths[idx[:split]], label_lengths[idx[split:]]

# MODEL

input_img = layers.Input(shape=(IMG_H, IMG_W, 1), name="image")

x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

# convert to sequence
new_shape = ((IMG_W // 8), (IMG_H // 8) * 128)
x = layers.Reshape(target_shape=new_shape)(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

output = layers.Dense(len(characters) + 1, activation="softmax")(x)

model = Model(input_img, output)

# CTC LOSS

def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], tf.int64)
    input_len = tf.cast(tf.shape(y_pred)[1], tf.int64)
    label_len = tf.cast(tf.math.count_nonzero(y_true, axis=-1), tf.int64)

    input_len = input_len * tf.ones(shape=(batch_len,1), dtype=tf.int64)
    label_len = label_len[:, tf.newaxis]

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=ctc_loss
)

model.summary()

# TRAIN

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50
)

model.save("captcha_ctc_model.keras")
