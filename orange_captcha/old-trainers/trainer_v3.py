import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

OUTPUT_DIR = "/home/jovyan/projects/captcha-reader/dataset/generated_samples"
IMG_W, IMG_H = 200, 50
BATCH_SIZE = 64
EPOCHS = 60
MODEL_SAVE_PATH = "/home/jovyan/projects/captcha-reader/captcha_pred_model_v3.keras"

print("GPU:", tf.config.list_physical_devices("GPU"))

characters = string.ascii_letters + string.digits + "@=#"
print("Charset size:", len(characters))

char_to_num = layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None,
    oov_token="[UNK]"
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)

paths = []
labels = []

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(OUTPUT_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))
assert len(paths) >= 100000

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return img[..., None]

print("Loading images...")
X = np.array([load_img(p) for p in paths], np.float32)

y = char_to_num(
    tf.strings.unicode_split(labels, "UTF-8")
).to_tensor(default_value=0)

y = y.numpy().astype(np.int32) - 1

y[y < 0] = 0

label_len = np.array([len(t) for t in labels], np.int32)

print("Label len min/max:", label_len.min(), label_len.max())
print("Sample encode:", labels[0], y[0][:label_len[0]])

idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(len(X)*0.9)

X_train, X_val = X[idx[:split]], X[idx[split:]]
y_train, y_val = y[idx[:split]], y[idx[split:]]
len_train, len_val = label_len[idx[:split]], label_len[idx[split:]]

labels_val = np.array(labels)[idx[split:]]

def make_ds(X,y,l):
    return (tf.data.Dataset
        .from_tensor_slices((X,y,l))
        .shuffle(20000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

train_ds = make_ds(X_train,y_train,len_train)
val_ds   = make_ds(X_val,y_val,len_val)

inp = layers.Input((IMG_H,IMG_W,1))

x = inp
for f in [64,128,256]:
    x = layers.Conv2D(f,3,padding="same",activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256,3,padding="same",activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)

x = layers.Dense(256,activation="relu")(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(
    256, return_sequences=True, unroll=True))(x)

x = layers.Bidirectional(layers.LSTM(
    128, return_sequences=True, unroll=True))(x)

out = layers.Dense(len(characters)+1,activation="softmax")(x)

base_model = models.Model(inp,out)
base_model.summary()

class CTCLoss(layers.Layer):
    def call(self, inputs):
        y_true,y_pred,label_len = inputs

        y_true = tf.cast(y_true, tf.int32)
        label_len = tf.cast(label_len, tf.int32)

        b = tf.shape(y_pred)[0]
        t = tf.shape(y_pred)[1]
        input_len = tf.ones((b,1),tf.int32)*t

        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len)

        tf.print("CTC loss:", tf.reduce_mean(loss))
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

labels_in = layers.Input((y.shape[1],),dtype="int32")
len_in = layers.Input((1,),dtype="int32")

loss_out = CTCLoss()([labels_in, base_model.output, len_in])

train_model = models.Model([inp,labels_in,len_in], loss_out)
train_model.compile(tf.keras.optimizers.Adam(1e-4))

def decode(pred):
    L = np.ones(pred.shape[0],np.int32)*pred.shape[1]
    dec,_ = tf.keras.backend.ctc_decode(pred,L,beam_width=10)
    return dec[0].numpy()

class Debug(callbacks.Callback):
    def on_epoch_end(self,e,l=None):
        p = base_model.predict(X_val[:8],verbose=0)
        d = decode(p)
        print("\nPreview:")
        for i,s in enumerate(d[:5]):
            txt = "".join(
                num_to_char(v+1).numpy().decode()
                for v in s if v!=-1)
            print("GT:", labels_val[i], "PR:", txt)

print("\nSanity overfit test on 512 samples")
small = make_ds(X_train[:512],y_train[:512],len_train[:512])
train_model.fit(
    small.map(lambda x,y,l:((x,y,l),None)),
    epochs=2)

train_model.fit(
    train_ds.map(lambda x,y,l:((x,y,l),None)),
    validation_data=val_ds.map(lambda x,y,l:((x,y,l),None)),
    epochs=EPOCHS,
    callbacks=[
        Debug(),
        callbacks.EarlyStopping(patience=8,restore_best_weights=True),
        callbacks.ModelCheckpoint("best.h5",
            save_weights_only=True,
            save_best_only=True)
])

base_model.save(MODEL_SAVE_PATH)
print("Saved:", MODEL_SAVE_PATH)
