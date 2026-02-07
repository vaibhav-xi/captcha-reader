import os
import cv2
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================= GPU SAFE SETUP =================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

print("\n===== ENV =====")
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("================\n")

# ================= CONFIG =================

OUTPUT_DIR = "/home/jovyan/projects/captcha-reader/dataset/generated_samples"
IMG_W, IMG_H = 200, 50
BATCH_SIZE = 64
EPOCHS = 60
MODEL_SAVE_PATH = "/home/jovyan/projects/captcha-reader/captcha_pred_model.keras"

characters = string.ascii_letters + string.digits + "@=#"
CHAR_LIST = list(characters)
NUM_CLASSES = len(CHAR_LIST) + 1

print("Charset size:", len(CHAR_LIST))
print("CTC classes:", NUM_CLASSES)

# ================= LOOKUP =================

char_to_num = layers.StringLookup(vocabulary=CHAR_LIST, mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)

# ================= LOAD PATHS =================

paths, labels = [], []

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".png"):
        paths.append(os.path.join(OUTPUT_DIR, f))
        labels.append(os.path.splitext(f)[0])

print("Samples:", len(paths))

# ================= IMAGE LOADER =================

def load_img(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return img[..., None]

print("Loading images to RAM...")
X = np.array([load_img(p) for p in paths], np.float32)
print("X shape:", X.shape)

# ================= LABEL ENCODE =================

split_chars = tf.strings.unicode_split(labels, "UTF-8")
y = char_to_num(split_chars).to_tensor(default_value=0)
y = tf.cast(y - 1, tf.int32)
y = tf.where(y < 0, 0, y)
y = y.numpy()

label_len = np.array([len(t) for t in labels], np.int32)

# ================= SPLIT =================

idx = np.arange(len(X))
np.random.shuffle(idx)
split = int(len(X)*0.9)

X_train, X_val = X[idx[:split]], X[idx[split:]]
y_train, y_val = y[idx[:split]], y[idx[split:]]
len_train, len_val = label_len[idx[:split]], label_len[idx[split:]]
labels_val = np.array(labels)[idx[split:]]

# ================= DATASET =================

def make_ds(X,y,l):
    ds = tf.data.Dataset.from_tensor_slices((X,y,l))
    ds = ds.shuffle(20000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds.map(lambda a,b,c: ((a,b,c), tf.zeros([tf.shape(a)[0]])))

train_ds = make_ds(X_train,y_train,len_train)
val_ds   = make_ds(X_val,y_val,len_val)

# ================= MODEL =================

inp = layers.Input((IMG_H,IMG_W,1))

x = inp
for f in [64,128,256]:
    x = layers.Conv2D(f,3,padding="same",activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,1))(x)

x = layers.Conv2D(256,3,padding="same",activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# ===== GRU SAFE MODE (NO cuDNN KERNEL) =====

x = layers.Bidirectional(
    layers.GRU(
        256,
        return_sequences=True,
        reset_after=False
    )
)(x)

x = layers.Bidirectional(
    layers.GRU(
        128,
        return_sequences=True,
        reset_after=False
    )
)(x)

out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

base_model = models.Model(inp,out)
base_model.summary()

# ================= CTC LOSS =================

class CTCLoss(layers.Layer):
    def call(self, inputs):
        y_true, y_pred, label_len = inputs

        b = tf.shape(y_pred)[0]
        t = tf.shape(y_pred)[1]

        input_len = tf.fill([b,1], t)
        label_len = tf.expand_dims(label_len, 1)

        loss = tf.keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_len,
            label_len
        )

        self.add_loss(tf.reduce_mean(loss))
        return y_pred

labels_in = layers.Input((y.shape[1],),dtype="int32")
len_in = layers.Input((),dtype="int32")

loss_out = CTCLoss()([labels_in, base_model.output, len_in])

train_model = models.Model([inp,labels_in,len_in], loss_out)
train_model.compile(tf.keras.optimizers.Adam(1e-4))

print("Train model ready")

# ================= DEBUG =================

def decode(pred):
    L = np.ones(pred.shape[0])*pred.shape[1]
    dec,_ = tf.keras.backend.ctc_decode(pred, L, greedy=True)
    return dec[0].numpy()

class Debug(callbacks.Callback):
    def on_epoch_end(self,e,l=None):
        p = base_model.predict(X_val[:8],verbose=0)
        d = decode(p)
        print("\nPreview epoch", e+1)
        for i,s in enumerate(d[:5]):
            txt = "".join(
                num_to_char(v).numpy().decode()
                for v in s if v >= 0 and v < len(CHAR_LIST))
            print("GT:", labels_val[i], "PR:", txt)

# ================= TRAIN =================

print("\nRunning sanity test")
small = make_ds(X_train[:512],y_train[:512],len_train[:512])
train_model.fit(small, epochs=2)

train_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        Debug(),
        callbacks.EarlyStopping(patience=8,restore_best_weights=True),
        callbacks.ModelCheckpoint(
            "best_weights.h5",
            save_weights_only=True,
            save_best_only=True)
    ])

base_model.save(MODEL_SAVE_PATH)
print("\nSaved:", MODEL_SAVE_PATH)
