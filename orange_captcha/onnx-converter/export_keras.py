import tensorflow as tf
from tensorflow import keras

m = keras.models.load_model("onnx_safe.keras", compile=False)

tf.saved_model.save(m, "onnx_safe_savedmodel")