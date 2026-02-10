import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

base = keras.models.load_model(
    "ocr_ctc_infer_safe_v9.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

feature_model = keras.Model(
    inputs=base.input,
    outputs=base.layers[-2].output
)

feature_model(tf.zeros([1,50,212,1]))

feature_model.export("saved_features_tfnet")

print("SavedModel exported")