import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

DIR_BASE   = "../../dataset/generated_samples_v10"
DIR_TARGET = "../../dataset/targeted_samples_v6"
DIR_REAL   = "../../dataset/orange-samples"
DIR_HARD   = "../../dataset/hard_negatives_v6"

START_MODEL = "ocr_ctc_onnx_safe.keras"
OUT_MODEL   = "ocr_ctc_onnx_safe_ft.keras"

WEIGHTS = {
    "base":   0.55,
    "target": 0.25,
    "hard":   0.15,
    "real":   0.05
}

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 6
STEPS_PER_EPOCH = 300

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

# ---------------- LOAD MODEL ----------------

print("Loading ONNX-safe model...")
model = keras.models.load_model(
    START_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)


# -------- freeze backbone, train head only --------

for l in model.layers:
    l.trainable = False

train_keywords = ["bidirectional", "tc", "dense", "head"]

for l in model.layers:
    name = l.name.lower()
    if any(k in name for k in train_keywords):
        l.trainable = True

print("Trainable layers:")
for l in model.layers:
    if l.trainable:
        print(" ", l.name)

print("Trainable params:",
      np.sum([np.prod(v.shape) for v in model.trainable_variables]))

# ---------------- DATA ----------------

def load_set(dirpath):
    if not os.path.exists(dirpath):
        return [],[],[]

    imgs,y,l = [],[],[]

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label = os.path.splitext(f)[0]
        if "_" in label:
            continue

        img = cv2.imread(os.path.join(dirpath,f))
        if img is None:
            continue

        enc=[characters.index(c) for c in label if c in characters]
        if not enc:
            continue

        imgs.append(img)
        y.append(enc)
        l.append(len(enc))

    return imgs,y,l


base_imgs,base_y,base_l     = load_set(DIR_BASE)
target_imgs,target_y,target_l = load_set(DIR_TARGET)
real_imgs,real_y,real_l     = load_set(DIR_REAL)
hard_imgs,hard_y,hard_l     = load_set(DIR_HARD)

print("Loaded:",
      len(base_imgs),
      len(target_imgs),
      len(real_imgs),
      len(hard_imgs))

POOLS = [
    (base_imgs,base_y,base_l),
    (target_imgs,target_y,target_l),
    (hard_imgs,hard_y,hard_l),
    (real_imgs,real_y,real_l)
]

P = np.array([
    WEIGHTS["base"],
    WEIGHTS["target"],
    WEIGHTS["hard"],
    WEIGHTS["real"]
])
P = P/P.sum()

# ---------------- PREPROCESS ----------------

def preprocess(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,(IMG_W,IMG_H))
    g=cv2.equalizeHist(g)
    g=cv2.copyMakeBorder(g,0,0,0,RIGHT_PAD,
                         cv2.BORDER_CONSTANT,value=255)
    return g.astype("float32")/255.0

def sample_source():
    while True:
        k=np.random.choice(len(POOLS),p=P)
        imgs,y,l = POOLS[k]
        if not imgs:
            continue
        i=np.random.randint(len(imgs))
        if l[i] <= 10:
            return imgs[i],y[i],l[i]

def gen_train():
    while True:
        img,y,l = sample_source()
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

# ---------------- CTC LOSS ----------------

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

        grads = tape.gradient(loss,self.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g,v in zip(grads,self.trainable_variables)]

        grads,_ = tf.clip_by_global_norm(grads,5.0)
        self.optimizer.apply_gradients(
            zip(grads,self.trainable_variables))

        gn=tf.linalg.global_norm(grads)
        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(model.inputs, model.outputs)

# ---------------- EVAL ----------------

def decode_batch(pred):
    L=np.ones(pred.shape[0])*pred.shape[1]
    d,_=tf.keras.backend.ctc_decode(pred,L,greedy=True)
    d=d[0].numpy()
    out=[]
    for s in d:
        s=s[s!=-1]
        out.append("".join(num_to_char(s+1).numpy().astype(str)))
    return out

def run_epoch_test(n=120):

    if not real_imgs:
        print("No real set — skipping eval")
        return

    idx=np.random.choice(len(real_imgs),min(n,len(real_imgs)))

    xs=[preprocess(real_imgs[i])[...,None] for i in idx]
    gt=["".join(num_to_char(np.array(real_y[i])+1).numpy().astype(str))
        for i in idx]

    p=model.predict(np.array(xs),verbose=0)
    pr=decode_batch(p)

    acc=sum(a==b for a,b in zip(pr,gt))/len(gt)

    print("Real decode acc:",round(acc*100,2))
    for i in range(min(5,len(gt))):
        print("GT:",gt[i],"| PR:",pr[i])

# ---------------- TRAIN ----------------

for e in range(EPOCHS):

    lr = 3e-6
    train_model.compile(
        optimizer=keras.optimizers.Adam(lr)
    )

    print("\n====================")
    print("FineTune Epoch",e+1,"lr",lr)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1,
        verbose=1
    )

    run_epoch_test()

model.save(OUT_MODEL)
print("\nSaved →", OUT_MODEL)
