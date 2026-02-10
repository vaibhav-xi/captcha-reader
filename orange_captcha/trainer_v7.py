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
EPOCHS = 18
STEPS_PER_EPOCH = 300

DIR_V2   = "../dataset/generated_samples_v4"
DIR_V3   = "../dataset/generated_samples_v5"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives_v4"

START_MODEL = "ocr_ctc_infer_safe_v6.keras"

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

def has_double_char(s):
    return any(a == b for a,b in zip(s,s[1:]))

def load_set(dirpath):
    paths, labels = [], []
    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath,f))
            labels.append(os.path.splitext(f)[0])

    imgs = [cv2.imread(p) for p in paths]

    y = char_to_num(
        tf.strings.unicode_split(labels,"UTF-8")
    ).to_tensor(default_value=-1)

    y = (y-1).numpy().astype(np.int32)

    lens = np.array([len(t) for t in labels],np.int32)
    dbl  = np.array([has_double_char(t) for t in labels])

    return imgs,y,lens,dbl

v2_imgs,v2_y,v2_l,v2_d = load_set(DIR_V2)
v3_imgs,v3_y,v3_l,v3_d = load_set(DIR_V3)
real_imgs,real_y,real_l,real_d = load_set(DIR_REAL)
hard_imgs,hard_y,hard_l,hard_d = load_set(DIR_HARD)

print("Loaded:",
      len(v2_imgs),len(v3_imgs),
      len(real_imgs),len(hard_imgs))

def preprocess(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(15,80,80),(40,255,255))
    mask = 255-mask
    mask = cv2.resize(mask,(IMG_W,IMG_H))
    mask = cv2.copyMakeBorder(
        mask,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )
    return mask.astype("float32")/255.0

def add_polygon_occluder(img, strength):
    if np.random.rand() > 0.3*strength:
        return img
    h,w = img.shape
    pts = np.array([
        [np.random.randint(w//3,w),0],
        [w,0],
        [w,h],
        [np.random.randint(w//3,w),h]
    ])
    cv2.fillPoly(img,[pts],0)
    return img

def augment(img,strength):
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g = add_polygon_occluder(g,strength)
    return preprocess(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR))

def pick(imgs,y,l,d):
    if np.random.rand()<0.35 and np.any(d):
        i=np.random.choice(np.where(d)[0])
    else:
        i=np.random.randint(len(imgs))
    return imgs[i],y[i],l[i]

def sample_source(stage):

    if stage == 0:
        P = (0.30,0.30,0.30,0.10)
    elif stage == 1:
        P = (0.25,0.25,0.30,0.20)
    else:
        P = (0.20,0.20,0.30,0.30)

    r=np.random.rand()
    if r<P[0]: return pick(v2_imgs,v2_y,v2_l,v2_d)
    if r<P[0]+P[1]: return pick(v3_imgs,v3_y,v3_l,v3_d)
    if r<P[0]+P[1]+P[2]: return pick(real_imgs,real_y,real_l,real_d)
    return pick(hard_imgs,hard_y,hard_l,hard_d)

epoch_var=tf.Variable(0)

def gen_train():
    while True:
        e=int(epoch_var.numpy())

        stage = 0 if e<6 else 1 if e<12 else 2
        strength = e/18

        img,y,l = sample_source(stage)

        if (y<0).any():
            continue

        yield augment(img,strength)[...,None], y, l

train_ds=tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W+RIGHT_PAD,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s=tf.shape(x)
    return tf.reshape(x,[s[0],s[1],s[2]*s[3]])

print("\nLoading base model...")

infer_model=tf.keras.models.load_model(
    START_MODEL,
    custom_objects={"collapse_hw":collapse_hw},
    compile=False
)

def set_stage(stage):

    for layer in infer_model.layers:
        layer.trainable=False

    if stage==0:
        infer_model.layers[-1].trainable=True

    elif stage==1:
        for l in infer_model.layers[-4:]:
            l.trainable=True

    else:
        for l in infer_model.layers:
            l.trainable=True

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
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))

        gn=tf.linalg.global_norm([gg for gg in g if gg is not None])

        return {
            "loss": loss,
            "grad_norm": gn
        }

train_model=CTCModel(
    infer_model.inputs,
    infer_model.outputs
)

def blank_stats(pred):
    return np.mean(pred.argmax(-1)==BLANK_IDX)

def decode_batch(pred):
    L=np.ones(pred.shape[0])*pred.shape[1]
    d,_=tf.keras.backend.ctc_decode(pred,L,greedy=True)
    d=d[0].numpy()
    out=[]
    for s in d:
        s=s[s!=-1]
        out.append("".join(num_to_char(s+1).numpy().astype(str)))
    return out

def run_epoch_test():
    idx=np.random.choice(len(real_imgs),80)
    xs=[preprocess(real_imgs[i])[...,None] for i in idx]
    gt=["".join(num_to_char(real_y[i]+1).numpy().astype(str)) for i in idx]
    p=infer_model.predict(np.array(xs),verbose=0)
    pr=decode_batch(p)
    acc=sum(a==b for a,b in zip(pr,gt))/len(gt)
    print("Blank %:",round(blank_stats(p)*100,2))
    print("Decode acc:",round(acc*100,2))
    for i in range(5):
        print("GT:",gt[i],"| PR:",pr[i])

for e in range(EPOCHS):

    epoch_var.assign(e)
    stage = 0 if e<6 else 1 if e<12 else 2

    lr = 1e-5 if stage==0 else 5e-6 if stage==1 else 2e-6

    print("\n====================")
    print("Epoch",e+1,"stage",stage,"lr",lr)

    set_stage(stage)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=5.0
        )
    )

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1
    )

    run_epoch_test()

infer_model.save("ocr_ctc_infer_safe_v8.keras")
print("\nSaved → ocr_ctc_infer_safe_v8.keras")
