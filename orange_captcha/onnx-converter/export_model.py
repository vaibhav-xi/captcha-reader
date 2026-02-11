import tensorflow as tf
from tensorflow import keras

MODEL = "ocr_ctc_no_lambda.keras"
OUT   = "ocr_ctc_savedmodel"

print("Loading patched model...")
m = keras.models.load_model(MODEL, compile=False)

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
