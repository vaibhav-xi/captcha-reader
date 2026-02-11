import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12
MAX_LABEL_LEN = 6

BATCH = 32
EPOCHS = 2
STEPS_PER_EPOCH = 800

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True,
    mask_token=None
)

def load_set(dirpath):
    imgs, labels = [], []
    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            imgs.append(cv2.imread(os.path.join(dirpath,f),0))
            labels.append(os.path.splitext(f)[0])

    y = char_to_num(
        tf.strings.unicode_split(labels,"UTF-8")
    ).to_tensor(default_value=0).numpy()

    y = np.maximum(y-1,0).astype(np.int32)
    lens = np.array([len(x) for x in labels], np.int32)

    y_fixed = np.zeros((len(y), MAX_LABEL_LEN), np.int32)
    for i,row in enumerate(y):
        y_fixed[i,:lens[i]] = row[:lens[i]]

    return imgs, y_fixed, lens

v2_imgs,v2_y,v2_l = load_set("../dataset/generated_samples_v2")

def preprocess(img):
    img = cv2.resize(img,(IMG_W,IMG_H))
    img = cv2.equalizeHist(img)
    img = cv2.copyMakeBorder(img,0,0,0,RIGHT_PAD,
                             cv2.BORDER_CONSTANT,255)
    return img.astype("float32")/255.0

def gen():
    while True:
        i = np.random.randint(len(v2_imgs))
        yield preprocess(v2_imgs[i])[...,None], v2_y[i], v2_l[i]

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W+RIGHT_PAD,1),tf.float32),
        tf.TensorSpec((MAX_LABEL_LEN,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH)

inp = layers.Input((IMG_H,IMG_W+RIGHT_PAD,1))

x = layers.Conv2D(32,3,activation="relu",padding="same")(inp)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Permute((2,1,3))(x)

time_steps = (IMG_W+RIGHT_PAD)//4
feat_dim = (IMG_H//4)*64

x = layers.Reshape((time_steps,feat_dim))(x)

def safe_lstm(units):
    return layers.LSTM(
        units,
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0.0
    )

x = layers.Bidirectional(safe_lstm(128))(x)
x = layers.Bidirectional(safe_lstm(64))(x)

out = layers.Dense(len(characters)+1,activation="softmax")(x)

model = models.Model(inp,out)

@tf.function
def ctc_loss(y_true,y_pred,label_len):
    b=tf.shape(y_pred)[0]
    t=tf.shape(y_pred)[1]
    inp_len=t*tf.ones((b,1),tf.int32)
    return tf.reduce_mean(
        tf.keras.backend.ctc_batch_cost(
            y_true,y_pred,inp_len,label_len[:,None]
        )
    )

class CTC(tf.keras.Model):
    def train_step(self,data):
        x,y,l=data
        with tf.GradientTape() as tape:
            p=self(x,training=True)
            loss=ctc_loss(y,p,l)

        g=tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))
        return {"loss":loss}

train_model = CTC(model.inputs, model.outputs)
train_model.compile(tf.keras.optimizers.Adam(1e-4))

for e in range(EPOCHS):
    print("Epoch",e+1)
    h=train_model.fit(ds,steps_per_epoch=STEPS_PER_EPOCH,epochs=1)

    if h.history["loss"][0] == 0:
        raise RuntimeError("CTC LOSS ZERO — aborting")

import tf2onnx

spec=(tf.TensorSpec(
    (None,IMG_H,IMG_W+RIGHT_PAD,1),
    tf.float32,name="image"),)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="ocr_safe.onnx"
)

print("ONNX saved")
