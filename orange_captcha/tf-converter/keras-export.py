import tensorflow as tf
from tensorflow import keras
import string

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

characters = string.ascii_letters + string.digits + "@=#"

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

print("Loading model...")

model = keras.models.load_model(
    "ocr_ctc_infer_safe_v9.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Model loaded")

@tf.function(
    input_signature=[
        tf.TensorSpec([None, IMG_H, IMG_W+RIGHT_PAD, 1], tf.float32)
    ]
)
def serve(x):
    return {"probs": model(x)}

print("Saving SavedModel...")

tf.saved_model.save(
    model,
    "captcha_savedmodel",
    signatures={"serving_default": serve}
)

print("Saved → captcha_savedmodel")
