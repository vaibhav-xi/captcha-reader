import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

DATASET_DIR = "../dataset/orange-samples"
MODEL_PATH  = "captcha_ctc_model_v2.keras"

IMG_W = 200
IMG_H = 50

BATCH_SIZE = 32
EPOCHS = 45
STEPS_PER_EPOCH = 400

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
)

paths = []
labels = []

for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(DATASET_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Real samples:", len(paths))

base_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
label_lens = np.array([len(t) for t in labels], dtype=np.int32)

y = char_to_num(
    tf.strings.unicode_split(labels, "UTF-8")
).to_tensor(default_value=0)

y = tf.maximum(y - 1, 0).numpy().astype(np.int32)

def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img

def aug_occlusion(img):
    if np.random.rand() < 0.9:
        h,w = img.shape
        x1 = np.random.randint(0,w)
        x2 = np.random.randint(0,w)
        y1 = np.random.randint(0,h)
        y2 = np.random.randint(0,h)
        thick = np.random.randint(12,28)
        cv2.line(img,(x1,y1),(x2,y2),0,thick)
    return img

def aug_vertical_bar(img):
    if np.random.rand() < 0.7:
        h,w = img.shape
        x = np.random.randint(0,w-8)
        cv2.rectangle(img,(x,0),(x+np.random.randint(8,22),h),0,-1)
    return img

def aug_diagonal_wedge(img):
    if np.random.rand() < 0.6:
        h,w = img.shape
        pts = np.array([
            [0,np.random.randint(0,h)],
            [w,np.random.randint(0,h)],
            [w,h],
            [0,h]
        ])
        cv2.fillPoly(img,[pts],0)
    return img

def aug_perspective(img):
    if np.random.rand() < 0.7:
        h,w = img.shape
        d = 18
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([
            [np.random.randint(-d,d),0],
            [w+np.random.randint(-d,d),0],
            [0,h],
            [w,h]
        ])
        M = cv2.getPerspectiveTransform(src,dst)
        img = cv2.warpPerspective(img,M,(w,h),borderValue=255)
    return img

def aug_morph(img):
    if np.random.rand() < 0.6:
        k = np.ones((2,2),np.uint8)
        img = cv2.erode(img,k) if np.random.rand()<0.5 else cv2.dilate(img,k)
    return img

def aug_contrast(img):
    a = np.random.uniform(0.6,1.4)
    b = np.random.uniform(-30,30)
    return np.clip(a*img+b,0,255).astype(np.uint8)

def augment(img):

    img = img.copy()

    img = aug_contrast(img)
    img = aug_perspective(img)
    img = aug_occlusion(img)
    img = aug_vertical_bar(img)
    img = aug_diagonal_wedge(img)
    img = aug_morph(img)

    if np.random.rand() < 0.5:
        c = np.random.randint(3,12)
        img = img[:,c:-c]
        img = cv2.resize(img,(IMG_W,IMG_H))

    return preprocess(img)

def gen():

    while True:

        idx = np.random.randint(0,len(base_imgs))
        img = augment(base_imgs[idx])
        yield img[...,None], y[idx], label_lens[idx]

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
)

ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class CTCModel(tf.keras.Model):

    def train_step(self,data):
        x,y,l = data
        with tf.GradientTape() as tape:
            p = self(x,training=True)
            loss = ctc_loss_fn(y,p,l)
        g = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(g,self.trainable_variables))
        return {"loss":tf.reduce_mean(loss)}

model = models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

for layer in model.layers[:-10]:
    layer.trainable = False

print("\nTrainable layers:")
for l in model.layers:
    if l.trainable:
        print(l.name)

@tf.function
def ctc_loss_fn(y_true,y_pred,label_len):

    b = tf.shape(y_pred)[0]
    t = tf.shape(y_pred)[1]
    inp_len = t*tf.ones((b,1),dtype=tf.int32)
    label_len = label_len[:,None]

    return tf.keras.backend.ctc_batch_cost(
        y_true,y_pred,inp_len,label_len
    )

ctc_model = CTCModel(model.inputs,model.outputs)

ctc_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5)
)

ctc_model.fit(
    ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH
)

ctc_model.save("captcha_ctc_adapted_v2.keras")

print("\nSaved → captcha_ctc_adapted_v2.keras")
