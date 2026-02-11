import os
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

START_MODEL = "ocr_ctc_infer_safe_v12b.keras"
OUT_MODEL   = "ocr_ctc_onnx_safe.keras"

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
EPOCHS = 18
STEPS_PER_EPOCH = 300

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)
NUM_CLASSES = len(characters) + 1

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s=tf.shape(x)
    return tf.reshape(x,[s[0],s[1],s[2]*s[3]])

base = keras.models.load_model(
    START_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Loaded base model")

dense = base.layers[-1]

feat = dense.input

onnx_head = layers.Conv1D(
    NUM_CLASSES,
    1,
    activation="linear",
    name="onnx_head"
)(feat)

infer_model = keras.Model(base.input, onnx_head)

for l in infer_model.layers:
    try:
        l.set_weights(base.get_layer(l.name).get_weights())
    except:
        pass

W,b = dense.get_weights()
infer_model.get_layer("onnx_head").set_weights([W[None,:,:], b])

print("Weights copied + head transferred")

for l in infer_model.layers:
    try:
        l.set_weights(base.get_layer(l.name).get_weights())
    except:
        pass
    
dense = base.layers[-1]
W,b = dense.get_weights()
infer_model.get_layer("onnx_head").set_weights([W[None,:,:], b])

print("Weights copied + head transferred")

def load_set(dirpath):

    if not os.path.exists(dirpath):
        return [],[],[]

    imgs,y,l=[],[],[]

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label=os.path.splitext(f)[0]
        if "_" in label:
            continue

        img=cv2.imread(os.path.join(dirpath,f))
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

if sum(len(p) for p in [base_imgs,target_imgs,real_imgs,hard_imgs]) == 0:
    raise RuntimeError("No dataset images found")

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

        loss = tf.where(tf.math.is_finite(loss), loss, 1e6)

        g=tape.gradient(loss,self.trainable_variables)
        g,_=tf.clip_by_global_norm(g,8.0)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))

        gn=tf.linalg.global_norm([gg for gg in g if gg is not None])
        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(infer_model.inputs, infer_model.outputs)

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

    p=infer_model.predict(np.array(xs),verbose=0)
    pr=decode_batch(p)

    acc=sum(a==b for a,b in zip(pr,gt))/len(gt)

    print("Real decode acc:",round(acc*100,2))
    for i in range(min(5,len(gt))):
        print("GT:",gt[i],"| PR:",pr[i])

def set_stage(stage):
    for l in infer_model.layers:
        l.trainable=False

    if stage==0:
        infer_model.get_layer("onnx_head").trainable=True
    elif stage==1:
        for l in infer_model.layers[-6:]:
            l.trainable=True
    else:
        for l in infer_model.layers:
            l.trainable=True

for e in range(EPOCHS):

    if e < 6:
        stage=0; lr=5e-5
    elif e < 12:
        stage=1; lr=5e-5
    else:
        stage=2; lr=1e-5

    set_stage(stage)

    train_model.compile(
        optimizer=keras.optimizers.Adam(lr)
    )

    print("\n====================")
    print("Epoch",e+1,"stage",stage,"lr",lr)

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1,
        verbose=1
    )

    run_epoch_test()

infer_model.save(OUT_MODEL)
print("\nSaved →", OUT_MODEL)
