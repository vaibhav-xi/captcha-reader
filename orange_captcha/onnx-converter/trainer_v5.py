import os
import cv2
import numpy as np
import string
import random
import tensorflow as tf
from tensorflow import keras

START_MODEL = "../ocr_ctc_onnx_safe_ft_v2.keras"
OUT_MODEL   = "../ocr_ctc_onnx_safe_ft_v6.keras"

DIRS = [
    "../../dataset/generated_samples_v12",
    "../../dataset/generated_samples_v13",
]

TEST_DIR = "../../dataset/test_samples"

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 12
STEPS_PER_EPOCH = 350

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = keras.layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

print("Loading ONNX-safe model...")
model = keras.models.load_model(
    START_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Model loaded")

def preprocess(img):

    if img.dtype != np.uint8:
        img = np.clip(img*255,0,255).astype(np.uint8)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g = cv2.resize(img,(IMG_W,IMG_H))
    g = cv2.equalizeHist(g)

    g = cv2.copyMakeBorder(
        g,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return g.astype("float32")/255.0

def load_dirs(dirs):

    xs, ys, ls = [], [], []

    for d in dirs:
        if not os.path.exists(d):
            continue

        for f in os.listdir(d):

            if not f.endswith(".png"):
                continue

            label = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(d,f), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            enc = [characters.index(c) for c in label if c in characters]
            if not enc:
                continue

            xs.append(img)
            ys.append(enc)
            ls.append(len(enc))

    return xs, ys, ls

imgs, labels, lengths = load_dirs(DIRS)
print("Loaded samples:", len(imgs))

hard_buffer = []

def sample_item():

    if hard_buffer and random.random() < 0.35:
        return random.choice(hard_buffer)

    i = random.randrange(len(imgs))
    return imgs[i], labels[i], lengths[i]

def gen_train():

    while True:
        img,y,l = sample_item()
        yield preprocess(img)[...,None], y, l

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
        self.optimizer.apply_gradients(
            zip(grads,self.trainable_variables))

        gn=tf.linalg.global_norm(grads)

        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(model.inputs, model.outputs)

def decode_with_conf(pred):

    best = np.argmax(pred,-1)
    probs = np.max(pred,-1)

    blank = pred.shape[-1]-1
    out_txt=[]
    out_conf=[]

    for seq,conf in zip(best,probs):

        chars=[]
        cs=[]
        prev=-1

        for s,c in zip(seq,conf):
            if s!=prev and s!=blank:
                chars.append(s)
                cs.append(c)
            prev=s

        if not chars:
            out_txt.append("")
            out_conf.append(0.0)
            continue

        txt=tf.strings.reduce_join(
            num_to_char(np.array(chars)+1)
        ).numpy().decode()

        out_txt.append(txt)
        out_conf.append(float(np.mean(cs)))

    return out_txt, out_conf

def load_test():

    xs,ys=[],[]

    for f in os.listdir(TEST_DIR):
        if not f.endswith(".png"):
            continue

        label=os.path.splitext(f)[0]
        img=cv2.imread(os.path.join(TEST_DIR,f),
                       cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        xs.append(preprocess(img)[...,None])
        ys.append(label)

    return np.array(xs), ys

test_x, test_y = load_test()
print("Test samples:", len(test_x))

for e in range(EPOCHS):

    lr = 3e-5 if e < 6 else 1e-5

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

    pred = model.predict(test_x,batch_size=64,verbose=0)
    txt, conf = decode_with_conf(pred)

    exact = sum(a==b for a,b in zip(test_y,txt))/len(test_y)

    print("Test exact acc:", round(exact,4))
    print("Mean confidence:", round(np.mean(conf),4))

    hard_buffer.clear()

    for i,c in enumerate(conf):

        if c >= 0.80:
            continue

        fname = test_y[i] + ".png"
        path  = os.path.join(TEST_DIR, fname)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        enc = [characters.index(ch) for ch in test_y[i]]

        hard_buffer.append((img, enc, len(enc)))

        print("Hard buffer size:", len(hard_buffer))

model.save(OUT_MODEL)
print("\nSaved →", OUT_MODEL)
