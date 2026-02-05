import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import string

from keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split

# CONFIG

DATASET_DIR = "dataset/samples"
IMG_H = 50
IMG_W = 200
MAX_LEN = 5

characters = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
nchar = len(characters)

# LOAD FILE LIST

files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".png")]
print("Total images:", len(files))


# PREPROCESS

def preprocess():
    X = []
    y = [[] for _ in range(MAX_LEN)]

    for pic in files:
        path = os.path.join(DATASET_DIR, pic)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # ensure consistent size
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, -1)

        label = os.path.splitext(pic)[0].lower()

        if len(label) != MAX_LEN:
            continue

        X.append(img)

        for i, ch in enumerate(label):
            vec = np.zeros(nchar)
            vec[characters.index(ch)] = 1
            y[i].append(vec)

    X = np.array(X)
    y = [np.array(pos) for pos in y]

    return X, y


# MODEL

def create_model():
    inp = layers.Input(shape=(IMG_H, IMG_W, 1))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = []
    for i in range(MAX_LEN):
        d = layers.Dense(128, activation="relu")(x)
        d = layers.Dropout(0.4)(d)
        outputs.append(
            layers.Dense(nchar, activation="softmax", name=f"char_{i}")(d)
        )

    model = Model(inp, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"] * MAX_LEN
    )

    return model

# DATA PREP

X, y = preprocess()

split = train_test_split(
    X, *y,
    test_size=0.2,
    random_state=42
)

X_train, X_test = split[0], split[1]
y_train = split[2::2]
y_test = split[3::2]

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# BUILD MODEL

model = create_model()
model.summary()


# CALLBACKS

cb = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint("captcha_model.h5", save_best_only=True)
]


# TRAIN

history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=cb
)


# PLOT LOSS

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["train", "val"])
plt.show()

# EVALUATE

print("\nTrain evaluation:")
model.evaluate(X_train, y_train)

print("\nTest evaluation:")
model.evaluate(X_test, y_test)

# PREDICT FUNCTION

def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found:", filepath)
        return None

    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    img = img[np.newaxis, :, :, np.newaxis]

    preds = model.predict(img, verbose=0)

    result = ""
    for p in preds:
        result += characters[np.argmax(p)]

    return result

# TEST PREDICTIONS

TEST_FILES = [
    "test_images/c1.png",
    "test_images/c2.png"
]

for f in TEST_FILES:
    if os.path.exists(f):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap="gray")
        plt.title(f)
        plt.show()
        print("Predicted:", predict(f))
