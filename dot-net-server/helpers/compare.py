import onnxruntime as ort
import numpy as np
import cv2
import tensorflow as tf
from keras import models

IMG_W = 200
IMG_H = 50

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img[None, ..., None]

img = preprocess("test1.png")

class CTCModel(tf.keras.Model):
    pass

keras_model = models.load_model(
    "captcha_ctc_adapted_v3.keras",
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

k_out = keras_model.predict(img)

sess = ort.InferenceSession("captcha_ctc.onnx")
o_out = sess.run(None, {"image": img})[0]

print("keras shape:", k_out.shape)
print("onnx shape:", o_out.shape)

print("keras max:", np.max(k_out))
print("onnx max:", np.max(o_out))

print("diff:", np.mean(np.abs(k_out - o_out)))
