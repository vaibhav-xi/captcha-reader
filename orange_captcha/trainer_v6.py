import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 24

BATCH = 32
EPOCHS = 18
STEPS_PER_EPOCH = 500

P_V2   = 0.20
P_V3   = 0.20
P_REAL = 0.40
P_HARD = 0.20

DIR_V2   = "../dataset/generated_samples_v2"
DIR_V3   = "../dataset/generated_samples_v3"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives"

START_MODEL = "ocr_ctc_infer_safe.keras"

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
    return any(a == b for a, b in zip(s, s[1:]))

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
    dbl  = np.array([has_double_char(t) for t in labels])

    return imgs, y, lens, dbl

v2_imgs, v2_y, v2_l, v2_d = load_set(DIR_V2)
v3_imgs, v3_y, v3_l, v3_d = load_set(DIR_V3)
real_imgs, real_y, real_l, real_d = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l, hard_d = load_set(DIR_HARD)

print("Loaded:",
      len(v2_imgs), len(v3_imgs),
      len(real_imgs), len(hard_imgs))

def preprocess(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (15,80,80), (40,255,255))
    mask = 255 - mask

    mask = cv2.resize(mask,(IMG_W,IMG_H))

    mask = cv2.copyMakeBorder(
        mask,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )

    return mask.astype("float32")/255.0

def add_polygon_occluder(img, strength):
    if np.random.rand() > 0.6 * strength:
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

def augment(img, strength):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = add_polygon_occluder(img, strength)
    return preprocess(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

def pick(imgs,y,l,dbl):
    if np.random.rand()<0.35 and np.any(dbl):
        i = np.random.choice(np.where(dbl)[0])
    else:
        i = np.random.randint(len(imgs))
    return imgs[i], y[i], l[i]

def sample_source():
    r=np.random.rand()
    if r<P_V2: return pick(v2_imgs,v2_y,v2_l,v2_d)
    if r<P_V2+P_V3: return pick(v3_imgs,v3_y,v3_l,v3_d)
    if r<P_V2+P_V3+P_REAL: return pick(real_imgs,real_y,real_l,real_d)
    return pick(hard_imgs,hard_y,hard_l,hard_d)

epoch_var=tf.Variable(0)

def gen_train():
    while True:
        img,y,l = sample_source()
        s=min(1.0,float(epoch_var.numpy())/10.0)
        yield augment(img,s)[...,None], y, l

train_ds=tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W+RIGHT_PAD,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

print("\nLoading base model...")
infer_model = tf.keras.models.load_model(
    START_MODEL,
    compile=False
)

for layer in infer_model.layers:
    layer.trainable = True

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
        return {"loss":loss,"grad_norm":gn}

train_model=CTCModel(
    infer_model.inputs,
    infer_model.outputs
)

train_model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5)
)

def blank_stats(pred):
    return np.mean(pred.argmax(-1)==BLANK_IDX)

def pred_entropy(pred):
    p=np.clip(pred,1e-8,1)
    return np.mean(-np.sum(p*np.log(p),axis=-1))

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

    print("Blank %:",round(blank_stats(p)*100,2))
    print("Entropy:",round(pred_entropy(p),3))

    pr=decode_batch(p)
    acc=sum(a==b for a,b in zip(pr,gt))/n
    print("Decode acc:",round(acc*100,2))

    for i in range(5):
        print("GT:",gt[i],"| PR:",pr[i])

for e in range(EPOCHS):
    epoch_var.assign(e)

    print("\n====================")
    print("FineTune Epoch",e+1,"/",EPOCHS)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1
    )

    run_epoch_test(infer_model,120)

infer_model.save("ocr_ctc_infer_safe_v6.keras")
print("\nSaved → ocr_ctc_infer_safe_v6.keras")
