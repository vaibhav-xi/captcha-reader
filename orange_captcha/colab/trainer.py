import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, mixed_precision

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

mixed_precision.set_global_policy("mixed_float16")


DATASET_DIR = "/dataset/generated_samples"

IMG_W = 200
IMG_H = 50
BATCH_SIZE = 32
EPOCHS = 20

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None,
    oov_token="[UNK]"
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))

idx = np.random.permutation(len(paths))
paths = np.array(paths)[idx]
labels = np.array(labels)[idx]

split = int(len(paths) * 0.8)

train_paths = paths[:split]
train_labels = labels[:split]

test_paths = paths[split:]
test_labels = labels[split:]

def load_image(path):
    path = path.numpy().decode()

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, -1)

    return img


def encode_sample(path, label):

    img = tf.py_function(load_image, [path], tf.float32)
    img.set_shape((IMG_H, IMG_W, 1))

    label = tf.strings.unicode_split(label, "UTF-8")
    label = char_to_num(label)

    label.set_shape([None])

    return img, label


def make_dataset(paths, labels):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            (IMG_H, IMG_W, 1),
            [None]
        ),
        padding_values=(
            0.0,
            tf.cast(0, tf.int64)
        )
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_paths, train_labels)
test_ds = make_dataset(test_paths, test_labels)

input_img = layers.Input(
    shape=(IMG_H, IMG_W, 1),
    name="image",
    dtype="float32"
)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(input_img)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)

# dynamic reshape
shape = keras.backend.int_shape(x)

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((shape[2], shape[1]*shape[3]))(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True)
)(x)

x = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True)
)(x)

output = layers.Dense(
    len(characters) + 1,
    activation="softmax",
    dtype="float32"
)(x)

model = keras.Model(input_img, output)

def ctc_loss(y_true, y_pred):

    batch = tf.shape(y_pred)[0]
    input_len = tf.shape(y_pred)[1]

    input_len = input_len * tf.ones((batch,1), dtype=tf.int32)

    label_len = tf.math.count_nonzero(y_true, axis=1, dtype=tf.int32)
    label_len = label_len[:,None]

    return keras.backend.ctc_batch_cost(
        y_true,
        y_pred,
        input_len,
        label_len
    )

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=ctc_loss
)

model.summary()

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

model.save("/content/captcha_ctc_model.keras")

print("Saved model")
