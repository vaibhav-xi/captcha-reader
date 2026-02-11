import os
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import layers, models

BASE_MODEL = "captcha_ctc_model_v2.keras"

ORANGE_DIR = "../dataset/orange_samples"
HARD_DIR   = "../dataset/hard_negatives"
BASE_DIR   = "../dataset/generated_samples_v3"

IMG_W = 200
IMG_H = 50

BATCH_SIZE = 32
EPOCHS = 50
STEPS_PER_EPOCH = 800

P_ORANGE = 0.5
P_HARD   = 0.3
P_BASE   = 0.2

characters = string.ascii_letters + string.digits + "@=#"

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None
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

orange_imgs, orange_y, orange_l = load_set(ORANGE_DIR)
hard_imgs,   hard_y,   hard_l   = load_set(HARD_DIR)
base_imgs,   base_y,   base_l   = load_set(BASE_DIR)

print("orange:", len(orange_imgs))
print("hard:",   len(hard_imgs))
print("base:",   len(base_imgs))

def preprocess(img):
    img = cv2.resize(img,(IMG_W,IMG_H))
    img = cv2.equalizeHist(img)
    return img.astype("float32")/255.0

def aug_bar(img):
    if np.random.rand()<0.7:
        h,w = img.shape
        y = np.random.randint(0,h)
        cv2.line(img,(0,y),(w,y+np.random.randint(-20,20)),0,
                 np.random.randint(12,30))

def aug_vertical_block(img):
    if np.random.rand()<0.6:
        h,w = img.shape
        x = np.random.randint(0,w-10)
        cv2.rectangle(img,(x,0),(x+np.random.randint(8,30),h),0,-1)

def aug_left_blackout(img):
    if np.random.rand()<0.4:
        h,w = img.shape
        cv2.rectangle(img,(0,0),(np.random.randint(10,50),h),0,-1)

def aug_top_cut(img):
    if np.random.rand()<0.5:
        h,w = img.shape
        cv2.rectangle(img,(0,0),(w,np.random.randint(8,25)),0,-1)

def aug_triangle(img):
    if np.random.rand()<0.5:
        h,w = img.shape
        pts = np.array([[0,0],[w,0],[0,h]])
        cv2.fillPoly(img,[pts],0)

def aug_perspective(img):
    if np.random.rand()<0.6:
        h,w = img.shape
        d=20
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([
            [np.random.randint(-d,d),0],
            [w+np.random.randint(-d,d),0],
            [0,h],
            [w,h]
        ])
        M = cv2.getPerspectiveTransform(src,dst)
        img[:] = cv2.warpPerspective(img,M,(w,h),borderValue=255)

def aug_morph(img):
    if np.random.rand()<0.5:
        k=np.ones((2,2),np.uint8)
        img[:] = cv2.erode(img,k) if np.random.rand()<0.5 else cv2.dilate(img,k)

def augment(img):

    img = img.copy()

    aug_bar(img)
    aug_vertical_block(img)
    aug_left_blackout(img)
    aug_top_cut(img)
    aug_triangle(img)
    aug_perspective(img)
    aug_morph(img)

    if np.random.rand()<0.5:
        c=np.random.randint(3,15)
        img = img[:,c:-c]
        img = cv2.resize(img,(IMG_W,IMG_H))

    return preprocess(img)

def sample_source():
    r = np.random.rand()
    if r < P_ORANGE:
        i = np.random.randint(len(orange_imgs))
        return orange_imgs[i], orange_y[i], orange_l[i]
    elif r < P_ORANGE + P_HARD and len(hard_imgs)>0:
        i = np.random.randint(len(hard_imgs))
        return hard_imgs[i], hard_y[i], hard_l[i]
    else:
        i = np.random.randint(len(base_imgs))
        return base_imgs[i], base_y[i], base_l[i]

def gen():
    while True:
        img,y,l = sample_source()
        yield augment(img)[...,None], y, l

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec((IMG_H,IMG_W,1),tf.float32),
        tf.TensorSpec((None,),tf.int32),
        tf.TensorSpec((),tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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

model = models.load_model(
    BASE_MODEL,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

for layer in model.layers[:-10]:
    layer.trainable = False

print("\nTrainable:")
for l in model.layers:
    if l.trainable:
        print(l.name)

ctc_model = CTCModel(model.inputs,model.outputs)

ctc_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5)
)

ctc_model.fit(
    ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH
)

ctc_model.save("captcha_ctc_adapted_v3.keras")
print("\nSaved captcha_ctc_adapted_v3.keras")
