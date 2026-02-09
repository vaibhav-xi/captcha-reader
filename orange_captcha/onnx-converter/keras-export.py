import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

base = keras.models.load_model(
    "ocr_ctc_infer_safe.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

feature_model = keras.Model(
    base.input,
    base.layers[-2].output
)

feature_model.export("saved_features")

print("Saved feature model")
