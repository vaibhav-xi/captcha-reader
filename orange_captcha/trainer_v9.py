import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 18
STEPS_PER_EPOCH = 300

DIR_V2   = "../dataset/generated_samples_v6"
DIR_V3   = "../dataset/targeted_images"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives_new"

START_MODEL = "ocr_ctc_infer_safe_v8.keras"

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

def load_set(dirpath):
    paths, labels = [], []

    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath, f))
            labels.append(os.path.splitext(f)[0])

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

v2_imgs, v2_y, v2_l = load_set(DIR_V2)
v3_imgs, v3_y, v3_l = load_set(DIR_V3)
real_imgs, real_y, real_l = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l = load_set(DIR_HARD)

print("Loaded:",
      len(v2_imgs), len(v3_imgs),
      len(real_imgs), len(hard_imgs))

def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g,(IMG_W,IMG_H))
    g = cv2.equalizeHist(g)

    g = cv2.copyMakeBorder(
        g,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return g.astype("float32")/255.0

epoch_var = tf.Variable(0)

def sample_source():

    e = int(epoch_var.numpy())

    # safer hard usage than v8
    if e < 6:
        pool = [(v2_imgs,v2_y,v2_l),
                (v3_imgs,v3_y,v3_l),
                (hard_imgs,hard_y,hard_l)]
        w = [0.60,0.30,0.10]

    elif e < 12:
        pool = [(v2_imgs,v2_y,v2_l),
                (v3_imgs,v3_y,v3_l),
                (hard_imgs,hard_y,hard_l),
                (real_imgs,real_y,real_l)]
        w = [0.40,0.25,0.20,0.15]

    else:
        pool = [(v2_imgs,v2_y,v2_l),
                (v3_imgs,v3_y,v3_l),
                (hard_imgs,hard_y,hard_l),
                (real_imgs,real_y,real_l)]
        w = [0.30,0.20,0.30,0.20]

    k = np.random.choice(len(pool), p=w)
    imgs,y,l = pool[k]

    i = np.random.randint(len(imgs))
    return imgs[i], y[i], l[i]

def gen_train():
    while True:
        img,y,l = sample_source()

        if l > 10:  # CTC safety
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

def set_trainable(stage):

    for layer in infer_model.layers:
        layer.trainable = False

    if stage == 0:
        for l in infer_model.layers[-2:]:
            l.trainable = True

    elif stage == 1:
        for l in infer_model.layers[-4:]:
            l.trainable = True

    else:
        for l in infer_model.layers:
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
        x,y,l=data
        with tf.GradientTape() as tape:
            p=self(x,training=True)
            loss=tf.reduce_mean(ctc_loss_fn(y,p,l))
        g=tape.gradient(loss,self.trainable_variables)

        g,_ = tf.clip_by_global_norm(g, 20.0)

        self.optimizer.apply_gradients(zip(g,self.trainable_variables))
        gn=tf.linalg.global_norm([gg for gg in g if gg is not None])
        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(
    infer_model.inputs,
    infer_model.outputs
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

def run_epoch_test(model,n=120):
    idx=np.random.choice(len(real_imgs),n)
    xs=[preprocess(real_imgs[i])[...,None] for i in idx]
    gt=["".join(num_to_char(real_y[i]+1).numpy().astype(str)) for i in idx]

    p=model.predict(np.array(xs),verbose=0)
    pr=decode_batch(p)

    acc=sum(a==b for a,b in zip(pr,gt))/n
    print("Decode acc:",round(acc*100,2))

    for i in range(5):
        print("GT:",gt[i],"| PR:",pr[i])

for e in range(EPOCHS):

    epoch_var.assign(e)

    if e < 8:
        stage = 0
        lr = 1e-5
    elif e < 14:
        stage = 1
        lr = 5e-6
    else:
        stage = 2
        lr = 2e-6

    set_trainable(stage)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    print("\n====================")
    print("Epoch",e+1,"stage",stage,"lr",lr)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1
    )

    run_epoch_test(infer_model,120)

infer_model.save("ocr_ctc_infer_safe_v9.keras")
print("\nSaved → ocr_ctc_infer_safe_v9.keras")
