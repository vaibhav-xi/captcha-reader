import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DIR_V10  = "../../dataset/generated_samples_v13"
DIR_V11  = "../../dataset/generated_samples_v12"
DIR_REAL = "../../dataset/orange-samples"
DIR_HARD = "../../dataset/hard_negatives_v6"

START_MODEL = "../ocr_ctc_onnx_safe_ft_v2.keras"
OUT_MODEL   = "../ocr_ctc_onnx_safe_ft_v5.keras"

WEIGHTS = {
    "v10":  0.55,
    "v11":  0.20,
    "real": 0.15,
    "hard": 0.10
}

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

BATCH = 32
EPOCHS = 3
STEPS_PER_EPOCH = 200

characters = string.ascii_letters + string.digits + "@=#"
BLANK_IDX = len(characters)

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x,[s[0],s[1],s[2]*s[3]])

print("Loading correction base model...")
model = keras.models.load_model(
    START_MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)
print("Model loaded")

def load_set(dirpath):
    imgs,y,l=[],[],[]

    if not os.path.exists(dirpath):
        return imgs,y,l

    for f in os.listdir(dirpath):
        if not f.endswith(".png"):
            continue

        label=os.path.splitext(f)[0]
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

v10_imgs,v10_y,v10_l = load_set(DIR_V10)
v11_imgs,v11_y,v11_l = load_set(DIR_V11)
real_imgs,real_y,real_l = load_set(DIR_REAL)
hard_imgs,hard_y,hard_l = load_set(DIR_HARD)

print("Loaded:",
      len(v10_imgs),
      len(v11_imgs),
      len(real_imgs),
      len(hard_imgs))

POOLS = [
    (v10_imgs,v10_y,v10_l),
    (v11_imgs,v11_y,v11_l),
    (real_imgs,real_y,real_l),
    (hard_imgs,hard_y,hard_l)
]

P = np.array([
    WEIGHTS["v10"],
    WEIGHTS["v11"],
    WEIGHTS["real"],
    WEIGHTS["hard"]
])
P = P/P.sum()

def preprocess(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,(IMG_W,IMG_H))
    g=cv2.equalizeHist(g)
    g=cv2.copyMakeBorder(
        g,0,0,0,RIGHT_PAD,
        cv2.BORDER_CONSTANT,value=255
    )
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

        grads=tape.gradient(loss,self.trainable_variables)

        grads=[
            tf.zeros_like(v) if g is None else g
            for g,v in zip(grads,self.trainable_variables)
        ]

        grads,_=tf.clip_by_global_norm(grads,5.0)
        self.optimizer.apply_gradients(
            zip(grads,self.trainable_variables)
        )

        gn=tf.linalg.global_norm(grads)

        return {"loss":loss,"grad_norm":gn}

train_model = CTCModel(model.inputs, model.outputs)

def quick_eval(n=150):

    if not real_imgs:
        print("No real set — skipping eval")
        return

    idx=np.random.choice(len(real_imgs),min(n,len(real_imgs)))

    xs=[preprocess(real_imgs[i])[...,None] for i in idx]
    gt=["".join(characters[c] for c in real_y[i])
        for i in idx]

    p=model.predict(np.array(xs),verbose=0)

    dec = tf.keras.backend.ctc_decode(
        p,
        np.ones(len(xs))*p.shape[1],
        greedy=True
    )[0][0].numpy()

    out=[]
    for s in dec:
        s=s[s!=-1]
        out.append("".join(characters[i] for i in s))

    acc=sum(a==b for a,b in zip(gt,out))/len(gt)

    print("Real exact acc:",round(acc*100,2))

    for i in range(min(5,len(gt))):
        print("GT:",gt[i],"| PR:",out[i])

for e in range(EPOCHS):

    lr = 3e-6

    train_model.compile(
        optimizer=keras.optimizers.Adam(lr)
    )

    print("\n====================")
    print("Epoch",e+1,"lr",lr)
    print("Trainable params:",
          np.sum([np.prod(v.shape)
          for v in train_model.trainable_variables]))

    train_model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=1,
        verbose=1
    )

    quick_eval()

model.save(OUT_MODEL)
print("\nSaved →", OUT_MODEL)
