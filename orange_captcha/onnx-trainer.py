import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

IMG_W = 200
IMG_H = 50
BATCH = 32
EPOCHS = 60
STEPS_PER_EPOCH = 800
VAL_STEPS = 120

DIR_V2 = "../dataset/generated_samples_v2"
DIR_V3 = "../dataset/generated_samples_v3"
DIR_REAL = "../dataset/orange-samples"
DIR_HARD = "../dataset/hard_negatives"

P_V2   = 0.30
P_V3   = 0.30
P_REAL = 0.25
P_HARD = 0.15

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

def load_set(dirpath):
    paths, labels = [], []
    for f in os.listdir(dirpath):
        if f.endswith(".png"):
            paths.append(os.path.join(dirpath,f))
            labels.append(os.path.splitext(f)[0])

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    lens = np.array([len(t) for t in labels], np.int32)

    y = char_to_num(
        tf.strings.unicode_split(labels,"UTF-8")
    ).to_tensor(default_value=0)

    y = tf.maximum(y-1,0).numpy().astype(np.int32)

    return imgs, y, lens

v2_imgs, v2_y, v2_l = load_set(DIR_V2)
v3_imgs, v3_y, v3_l = load_set(DIR_V3)
real_imgs, real_y, real_l = load_set(DIR_REAL)
hard_imgs, hard_y, hard_l = load_set(DIR_HARD)

print("Loaded sets:",
      len(v2_imgs), len(v3_imgs),
      len(real_imgs), len(hard_imgs))

def preprocess(img):
    img = cv2.resize(img,(IMG_W,IMG_H))
    img = cv2.equalizeHist(img)
    return img.astype("float32")/255.0

def augment(img, strength):

    img = img.copy()

    if np.random.rand() < 0.20:
        return preprocess(img)

    if np.random.rand() < 0.3*strength:
        h,w = img.shape
        x = np.random.randint(0,w-12)
        cv2.rectangle(img,(x,0),(x+np.random.randint(8,24),h),0,-1)

    if np.random.rand() < 0.4*strength:
        h,w = img.shape
        y = np.random.randint(0,h)
        cv2.line(img,(0,y),(w,y+np.random.randint(-20,20)),0,
                 np.random.randint(8,22))

    if np.random.rand() < 0.3*strength:
        k = np.ones((2,2),np.uint8)
        img = cv2.erode(img,k) if np.random.rand()<0.5 else cv2.dilate(img,k)

    if np.random.rand() < 0.3*strength:
        h,w = img.shape
        d=18
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([
            [np.random.randint(-d,d),0],
            [w+np.random.randint(-d,d),0],
            [0,h],
            [w,h]
        ])
        M = cv2.getPerspectiveTransform(src,dst)
        img = cv2.warpPerspective(img,M,(w,h),borderValue=255)

    return preprocess(img)

def sample_source():
    r = np.random.rand()

    if r < P_V2:
        i = np.random.randint(len(v2_imgs))
        return v2_imgs[i], v2_y[i], v2_l[i]

    if r < P_V2 + P_V3:
        i = np.random.randint(len(v3_imgs))
        return v3_imgs[i], v3_y[i], v3_l[i]

    if r < P_V2 + P_V3 + P_REAL:
        i = np.random.randint(len(real_imgs))
        return real_imgs[i], real_y[i], real_l[i]

    i = np.random.randint(len(hard_imgs))
    return hard_imgs[i], hard_y[i], hard_l[i]

epoch_var = tf.Variable(0, dtype=tf.int32)

def gen_train():
    while True:
        img,y,l = sample_source()
        strength = min(1.0, float(epoch_var.numpy())/30.0)
        yield augment(img,strength)[...,None], y, l

def gen_val():
    while True:
        i = np.random.randint(len(real_imgs))
        yield preprocess(real_imgs[i])[...,None], real_y[i], real_l[i]

train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    gen_val,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH)

inp = layers.Input(shape=(IMG_H,IMG_W,1))

x = layers.Conv2D(32,3,activation="relu",padding="same")(inp)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D((2,1))(x)

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((IMG_W//4, 6*128))(x)

x = layers.Dense(128,activation="relu")(x)
x = layers.Bidirectional(layers.LSTM(
    128,
    return_sequences=True,
    unroll=False
))(x)

x = layers.Bidirectional(layers.LSTM(
    64,
    return_sequences=True,
    unroll=False
))(x)

out = layers.Dense(len(characters)+1,activation="softmax")(x)

infer_model = models.Model(inp,out)

@tf.function
def ctc_loss_fn(y_true,y_pred,label_len):
    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t*tf.ones((b,1),tf.int32)
    label_len = label_len[:,None]
    return tf.keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len)

class CTCModel(tf.keras.Model):
    def train_step(self,data):
        x,y,l = data
        with tf.GradientTape() as tape:
            p = self(x,training=True)
            loss = ctc_loss_fn(y,p,l)
        g = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))
        return {"loss":tf.reduce_mean(loss)}

    def test_step(self,data):
        x,y,l = data
        p = self(x,training=False)
        loss = ctc_loss_fn(y,p,l)
        return {"loss":tf.reduce_mean(loss)}

train_model = CTCModel(infer_model.inputs, infer_model.outputs)

lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[20*STEPS_PER_EPOCH, 40*STEPS_PER_EPOCH],
    values=[1e-4, 5e-5, 2e-5]
)

train_model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_sched)
)

def decode_batch(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded,_ = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )

    decoded = decoded[0].numpy()

    texts = []
    for seq in decoded:
        seq = seq[seq != -1]
        chars = num_to_char(seq+1).numpy().astype(str)
        texts.append("".join(chars))

    return texts


def run_epoch_test(model, samples=120):

    idxs = np.random.choice(len(real_imgs), samples, replace=True)

    imgs=[]
    gt=[]

    for i in idxs:
        imgs.append(preprocess(real_imgs[i])[...,None])
        gt.append("".join(
            num_to_char(real_y[i]+1).numpy().astype(str)
        ))

    x=np.array(imgs)
    pred=model.predict(x,verbose=0)
    pr=decode_batch(pred)

    correct=sum(p==t for p,t in zip(pr,gt))
    acc=correct/samples

    print("\nDecode accuracy:", round(acc*100,2),"%")

    for i in range(5):
        print("GT:",gt[i]," | PR:",pr[i])

for e in range(EPOCHS):
    epoch_var.assign(e)

    print("\n========================")
    print("Epoch",e+1,"/",EPOCHS)

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        epochs=1
    )

    run_epoch_test(infer_model,120)

infer_model.save("ocr_ctc_infer.keras")

import tf2onnx

spec=(tf.TensorSpec((None,IMG_H,IMG_W,1),tf.float32,name="image"),)

tf2onnx.convert.from_keras(
    infer_model,
    input_signature=spec,
    opset=13,
    output_path="ocr_ctc.onnx"
)

print("\nSaved → ocr_ctc.onnx")
