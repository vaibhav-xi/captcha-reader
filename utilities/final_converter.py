import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from keras import layers, models

SRC_MODEL = "captcha_ctc_adapted_v3.keras"
ONNX_OUT = "captcha_ctc.onnx"

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        pass

print("Loading trained model...")
src = tf.keras.models.load_model(
    SRC_MODEL,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("Saving temp weights...")
src.save_weights("tmp.weights.h5")

print("Building ONNX-safe model...")

inputs = layers.Input(shape=(50,200,1), name="image")

x = layers.Conv2D(32,3,padding="same",activation="relu")(inputs)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64,3,padding="same",activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128,3,padding="same",activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,1))(x)

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((50,768))(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Bidirectional(
    layers.LSTM(
        128,
        return_sequences=True,
        activation="tanh",
        unroll=True
    )
)(x)

x = layers.Bidirectional(
    layers.LSTM(
        64,
        return_sequences=True,
        activation="tanh",
        unroll=True
    )
)(x)

outputs = layers.Dense(66, activation="softmax")(x)

model = models.Model(inputs, outputs)

print("Loading weights...")
model.load_weights("tmp.weights.h5")

_ = model(tf.zeros([1,50,200,1]))

print("Converting to ONNX...")

spec = (tf.TensorSpec((None,50,200,1), tf.float32, name="image"),)

onnx_model,_ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17
)

open(ONNX_OUT,"wb").write(
    onnx_model.SerializeToString()
)

print("Validating ONNX...")
ort.InferenceSession(ONNX_OUT)

print("SUCCESS — ONNX model valid")
