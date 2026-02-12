import os
import cv2
import numpy as np
import string
import random
import tensorflow as tf
from tensorflow import keras

TRAIN_DIR = "../../dataset/generated_samples_v14"
TEST_DIR  = "../../dataset/test_latest"

MODEL_IN  = "../ocr_ctc_onnx_safe_ft_v6.keras"
MODEL_OUT = "../ocr_ctc_onnx_safe_ft_v7.keras"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 8
STEPS_PER_EPOCH = 400

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

CONFUSION_SETS = [
    "lI1iJj",
    "gq",
    "mn",
    "OQ0",
    "vy",
    "B8",
    "S5",
    "Z2",
]

CONF_CHARS = set("".join(CONFUSION_SETS))

def preprocess(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(IMG_W,IMG_H))
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)

    img = cv2.copyMakeBorder(
        img,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return img.astype("float32")/255.0

def load_set(dirpath):
    imgs, ys, ls = [], [], []

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label = os.path.splitext(f)[0]
        path = os.path.join(dirpath,f)

        img = cv2.imread(path)
        if img is None:
            continue

        enc = [characters.index(c) for c in label if c in characters]
        if not enc:
            continue

        imgs.append(img)
        ys.append(enc)
        ls.append(len(enc))

    return imgs, ys, ls


print("Loading train set...")
tr_imgs, tr_y, tr_l = load_set(TRAIN_DIR)
print("Train samples:", len(tr_imgs))

print("Loading test set...")
te_imgs, te_y, te_l = load_set(TEST_DIR)
print("Test samples:", len(te_imgs))

def label_weight(lbl):
    s = "".join(characters[i] for i in lbl)

    w = 1.0

    if any(c in CONF_CHARS for c in s):
        w *= 2.5

    if any(s[i]==s[i+1] for i in range(len(s)-1)):
        w *= 2.0

    if any(c in "ilIjJ" for c in s):
        w *= 1.8

    return w


weights = np.array([label_weight(y) for y in tr_y])
weights /= weights.sum()


def gen_train():
    while True:
        i = np.random.choice(len(tr_imgs), p=weights)
        yield preprocess(tr_imgs[i])[...,None], tr_y[i], tr_l[i]


train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W+RIGHT_PAD,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).padded_batch(
    BATCH,
    padded_shapes=((IMG_H,IMG_W+RIGHT_PAD,1),(None,),()),
    padding_values=(0.0, BLANK_IDX, 0)
).prefetch(tf.data.AUTOTUNE)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

print("Loading ONNX-safe model...")
model = keras.models.load_model(
    MODEL_IN,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)
print("Model loaded")

def ctc_loss(y_true,y_pred,label_len):
    b=tf.shape(y_pred)[0]
    t=tf.shape(y_pred)[1]
    inp_len=t*tf.ones((b,1),tf.int32)
    label_len=label_len[:,None]
    return tf.keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len)


class CTCModel(keras.Model):
    def train_step(self,data):
        x,y,l=data
        with tf.GradientTape() as tape:
            p=self(x,training=True)
            loss=tf.reduce_mean(ctc_loss(y,p,l))

        grads=tape.gradient(loss,self.trainable_variables)
        grads=[tf.zeros_like(v) if g is None else g
               for g,v in zip(grads,self.trainable_variables)]
        grads,_=tf.clip_by_global_norm(grads,8.0)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        gn=tf.linalg.global_norm(grads)
        return {"loss":loss,"grad_norm":gn}


train_model = CTCModel(model.inputs, model.outputs)

def quick_test():
    xs = np.array([preprocess(im)[...,None] for im in te_imgs[:200]])
    p  = model.predict(xs,verbose=0)

    best = p.argmax(-1)
    conf = p.max(-1).mean()

    print("Test mean confidence:", round(float(conf),4))

for e in range(EPOCHS):

    lr = 3e-5 if e < 4 else 1e-5

    train_model.compile(
        optimizer=keras.optimizers.Adam(lr)
    )

    print("\n====================")
    print("Epoch",e+1,"lr",lr)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1,
        verbose=1
    )

    quick_test()


model.save(MODEL_OUT)
print("\nSaved →", MODEL_OUT)
