import cv2
import numpy as np
import tensorflow as tf
from keras import models
import onnxruntime as ort

IMG_W = 200
IMG_H = 50

MODEL_PATH = "captcha_ctc_adapted_v3.keras"
ONNX_PATH  = "captcha_ctc_fixed.onnx"

IMAGE_PATH = "test1.png"

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img


img = preprocess(IMAGE_PATH)
batch = img[None, ..., None]

print("pixel sample:", img[0,0], img[0,1], img[0,2])

class CTCModel(tf.keras.Model):
    pass

tf_model = models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

tf_out = tf_model.predict(batch, verbose=0)

print("\nTF output shape:", tf_out.shape)

tf_top5_idx = np.argsort(tf_out[0,0])[-5:]
tf_top5_val = np.sort(tf_out[0,0])[-5:]

print("TF top5 idx :", tf_top5_idx)
print("TF top5 prob:", tf_top5_val)
print("TF argmax   :", np.argmax(tf_out[0,0]))

sess = ort.InferenceSession(ONNX_PATH)

onnx_out = sess.run(None, {"image": batch.astype(np.float32)})[0]

print("\nONNX output shape:", onnx_out.shape)

ox_top5_idx = np.argsort(onnx_out[0,0])[-5:]
ox_top5_val = np.sort(onnx_out[0,0])[-5:]

print("ONNX top5 idx :", ox_top5_idx)
print("ONNX top5 prob:", ox_top5_val)
print("ONNX argmax   :", np.argmax(onnx_out[0,0]))
