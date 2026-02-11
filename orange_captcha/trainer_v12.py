import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TMPDIR"] = "/Volumes/samsung_980/ml_cache/tmp"

import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 6
STEPS_PER_EPOCH = 250

DIR_BASE = "../dataset/generated_samples_v9"
DIR_TARGET = "../dataset/targeted_samples_v5"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives_new"

START_MODEL = "models/ocr_ctc_infer_safe_v11.keras"

W_BASE = 0.50
W_TARGET = 0.35
W_HARD = 0.10
W_REAL = 0.05

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

def clean_label(name):
    name = name.split("_")[0]
    return name

def load_set(dirpath):

    paths, labels = [], []

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue
        label = clean_label(os.path.splitext(f)[0])
        paths.append(os.path.join(dirpath,f))
        labels.append(label)

    imgs = [cv2.imread(p) for p in paths]

    y = char_to_num(
        tf.strings.unicode_split(labels, "UTF-8")
    ).to_tensor(default_value=-1)

    y = (y - 1).numpy().astype(np.int32)
    lens = np.array([len(t) for t in labels], np.int32)

    good = np.all(y >= 0, axis=1)

    return (
        [imgs[i] for i in range(len(imgs)) if good[i]],
        y[good],
        lens[good],
    )

base_imgs, base_y, base_l = load_set(DIR_BASE)
target_imgs, target_y, target_l = load_set(DIR_TARGET)
real_imgs, real_y, real_l = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l = load_set(DIR_HARD)

print("Loaded:",
      len(base_imgs),
      len(target_imgs),
      len(real_imgs),
      len(hard_imgs))

def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g,(IMG_W,IMG_H))
    g = cv2.equalizeHist(g)

    g = cv2.copyMakeBorder(
        g,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return g.astype("float32")/255.0

pools = [
    (base_imgs, base_y, base_l, W_BASE),
    (target_imgs, target_y, target_l, W_TARGET),
    (hard_imgs, hard_y, hard_l, W_HARD),
    (real_imgs, real_y, real_l, W_REAL),
]

weights = np.array([p[3] for p in pools])
weights = weights / weights.sum()

def sample_source():

    k = np.random.choice(len(pools), p=weights)
    imgs,y,l,_ = pools[k]

    if len(imgs) == 0:
        return sample_source()

    i = np.random.randint(len(imgs))
    return imgs[i], y[i], l[i]

def gen_train():
    while True:

        img,y,l = sample_source()

        if l > 10:
            continue

        yield preprocess(img)[...,None], y, l

train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W+RIGHT_PAD,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(t):
    s = tf.shape(t)
    return tf.reshape(t, [s[0], s[1], s[2]*s[3]])

infer_model = tf.keras.models.load_model(
    START_MODEL,
    custom_objects={"collapse_hw": collapse_hw},
    compile=False
)

for layer in infer_model.layers:
    layer.trainable = False

for l in infer_model.layers[-4:]:
    l.trainable = True

@tf.function
def ctc_loss_fn(y_true,y_pred,label_len):
    b=tf.shape(y_pred)[0]
    t=tf.shape(y_pred)[1]
    inp_len=t*tf.ones((b,1),tf.int32)
    label_len=label_len[:,None]
    return tf.keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len)

class CTCModel(tf.keras.Model):

    def train_step(self,data):

        x,y,l = data

        with tf.GradientTape() as tape:
            p = self(x,training=True)
            loss = tf.reduce_mean(ctc_loss_fn(y,p,l))

        g = tape.gradient(loss,self.trainable_variables)
        g,_ = tf.clip_by_global_norm(g, 6.0)

        self.optimizer.apply_gradients(zip(g,self.trainable_variables))

        gn = tf.linalg.global_norm([gg for gg in g if gg is not None])

        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(
    infer_model.inputs,
    infer_model.outputs
)

train_model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-6)
)

def decode_batch(pred):
    L=np.ones(pred.shape[0])*pred.shape[1]
    d,_=tf.keras.backend.ctc_decode(pred,L,greedy=True)
    d=d[0].numpy()
    out=[]
    for s in d:
        s=s[s!=-1]
        out.append("".join(num_to_char(s+1).numpy().astype(str)))
    return out

def run_real_test(n=120):

    idx=np.random.choice(len(real_imgs),n)
    xs=[preprocess(real_imgs[i])[...,None] for i in idx]
    gt=["".join(num_to_char(real_y[i]+1).numpy().astype(str)) for i in idx]

    p=infer_model.predict(np.array(xs),verbose=0)
    pr=decode_batch(p)

    acc=sum(a==b for a,b in zip(pr,gt))/n
    print("Real decode acc:",round(acc*100,2))

for e in range(EPOCHS):

    print("\n====================")
    print("Micro Epoch", e+1, "/", EPOCHS)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1
    )

    run_real_test()

infer_model.save("ocr_ctc_infer_safe_v12.keras")

print("\nSaved → ocr_ctc_infer_safe_v12.keras")
