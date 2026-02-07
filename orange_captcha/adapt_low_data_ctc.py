import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

DATASET_DIR = "../dataset/orange-samples"
MODEL_PATH  = "captcha_ctc_model_v2.keras"

IMG_W = 200
IMG_H = 50

AUG_PER_IMAGE = 120
BATCH_SIZE = 32
EPOCHS = 20

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

def add_occlusion(img):
    h, w = img.shape
    if np.random.rand() < 0.7:
        x1 = np.random.randint(0, w//2)
        x2 = np.random.randint(w//2, w)
        y = np.random.randint(0, h)
        thickness = np.random.randint(8, 18)
        cv2.line(img, (x1,y), (x2,y+np.random.randint(-10,10)), 0, thickness)
    return img


def add_diagonal_mask(img):
    if np.random.rand() < 0.5:
        h, w = img.shape
        pts = np.array([
            [0, np.random.randint(0,h)],
            [w, np.random.randint(0,h)],
            [w, h],
            [0, h]
        ])
        cv2.fillPoly(img, [pts], 0)
    return img


def perspective_warp(img):
    if np.random.rand() < 0.6:
        h, w = img.shape
        dx = np.random.randint(-15, 15)
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([[dx,0],[w-dx,0],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w,h), borderValue=255)
    return img


def morph_jitter(img):
    if np.random.rand() < 0.5:
        k = np.ones((2,2),np.uint8)
        if np.random.rand() < 0.5:
            img = cv2.erode(img, k)
        else:
            img = cv2.dilate(img, k)
    return img


def contrast_jitter(img):
    alpha = np.random.uniform(0.7, 1.3)
    beta  = np.random.uniform(-20, 20)
    img = np.clip(alpha*img + beta, 0, 255).astype(np.uint8)
    return img


def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img


def augment(img):

    img = contrast_jitter(img)
    img = perspective_warp(img)
    img = add_occlusion(img)
    img = add_diagonal_mask(img)
    img = morph_jitter(img)

    if np.random.rand() < 0.5:
        crop = np.random.randint(2,10)
        img = img[:, crop:-crop]
        img = cv2.resize(img, (IMG_W, IMG_H))

    return img

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Real samples:", len(paths))

base_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

X = []
Y = []
L = []

for img, label in zip(base_imgs, labels):

    for _ in range(AUG_PER_IMAGE):

        aug = augment(img.copy())
        aug = preprocess(aug)

        X.append(aug)
        Y.append(label)
        L.append(len(label))

X = np.array(X)[..., np.newaxis]

y = char_to_num(
    tf.strings.unicode_split(Y, "UTF-8")
).to_tensor(default_value=0)

y = tf.maximum(y - 1, 0).numpy().astype(np.int32)
L = np.array(L, dtype=np.int32)

print("Augmented samples:", len(X))

ds = tf.data.Dataset.from_tensor_slices((X,y,L))
ds = ds.shuffle(10000).batch(BATCH_SIZE)

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

model = models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("\nLoaded base model")

for layer in model.layers[:-6]:
    layer.trainable = False

@tf.function
def ctc_loss_fn(y_true, y_pred, label_len):

    batch = tf.shape(y_pred)[0]
    input_len = tf.shape(y_pred)[1]
    input_len = input_len * tf.ones((batch,1), dtype=tf.int32)
    label_len = label_len[:,None]

    return tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_len, label_len
    )

ctc_model = CTCModel(model.inputs, model.outputs)

ctc_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5)
)

ctc_model.fit(ds, epochs=EPOCHS)

ctc_model.save("captcha_ctc_adapted.keras")

print("\nSaved adapted model → captcha_ctc_adapted.keras")
