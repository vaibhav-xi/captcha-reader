import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import string
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split

# ======================
# CONFIG
# ======================
DATASET_DIR = "dataset/samples"
IMG_H, IMG_W = 50, 200
MAX_LEN = 5

characters = string.ascii_lowercase + "0123456789"
nchar = len(characters)

# ======================
# LOAD FILENAMES
# ======================
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png")]
n = len(files)

print("Total images:", n)

# ======================
# PREPROCESS
# ======================
def preprocess():
    X = []
    y = [[] for _ in range(MAX_LEN)]

    for pic in files:
        path = os.path.join(DATASET_DIR, pic)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize for safety (important improvement)
        img = cv2.resize(img, (IMG_W, IMG_H))

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, -1)

        label = os.path.splitext(pic)[0].lower()

        if len(label) != MAX_LEN:
            continue

        X.append(img)

        for i, ch in enumerate(label):
            onehot = np.zeros(nchar)
            onehot[characters.index(ch)] = 1
            y[i].append(onehot)

    X = np.array(X)
    y = [np.array(pos) for pos in y]

    return X, y


# ======================
# MODEL
# ======================
def create_model():
    inp = layers.Input(shape=(IMG_H, IMG_W, 1))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    outputs = []
    for i in range(MAX_LEN):
        d = layers.Dense(128, activation="relu")(x)
        d = layers.Dropout(0.4)(d)
        outputs.append(layers.Dense(nchar, activation="softmax", name=f"char_{i}")(d))

    model = Model(inp, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ======================
# TRAIN
# ======================
X, y = preprocess()

X_train, X_test, *y_split = train_test_split(
    X, *y, test_size=0.2, random_state=42
)

y_train = y_split[:MAX_LEN]
y_test = y_split[MAX_LEN:]

model = create_model()
model.summary()

cb = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint("captcha_model.h5", save_best_only=True)
]

history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=cb
)

# ======================
# PLOTS
# ======================
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["train", "val"])
plt.show()

# ======================
# EVALUATE
# ======================
print("\nTrain:")
model.evaluate(X_train, y_train)

print("\nTest:")
model.evaluate(X_test, y_test)


# ======================
# PREDICT
# ======================
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found")
        return None

    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    img = img[np.newaxis, :, :, np.newaxis]

    preds = model.predict(img)

    result = ""
    for p in preds:
        result += characters[np.argmax(p)]

    return result


# ======================
# TEST LOCAL FILES
# ======================
for f in ["test_images/mnop2.png", "test_images/34d.png"]:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap="gray")
    plt.show()
    print("Predicted:", predict(f))
