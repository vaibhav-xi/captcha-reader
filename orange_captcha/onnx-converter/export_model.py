import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

MODEL = "ocr_ctc_onnx_safe.keras"
OUT   = "ocr_ctc_savedmodel"

print("Loading patched model...")
m = keras.models.load_model(
    MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Building serving function...")

@tf.function(
    input_signature=[
        tf.TensorSpec([None,50,212,1], tf.float32)
    ]
)
def serve(x):
    return {"probs": m(x)}

print("Saving SavedModel...")
tf.saved_model.save(
    m,
    OUT,
    signatures={"serving_default": serve}
)

print("Saved: ", OUT)
