import tensorflow as tf
import tf2onnx
from tensorflow import keras

########################################
# needed to load original Lambda model
########################################

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2] * s[3]])

########################################
# ONNX-safe replacement layer
########################################

class CollapseHWStatic(tf.keras.layers.Layer):
    def call(self, x, mask=None, training=None):
        b, h, w, c = x.shape
        return tf.reshape(x, (-1, h, w * c))

########################################

print("Loading model...")

model = keras.models.load_model(
    "ocr_ctc_infer_safe.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

########################################
# clone graph safely
########################################

def replace_layer(layer):
    if isinstance(layer, tf.keras.layers.Lambda) and layer.name == "collapse_hw":
        print("Replacing collapse_hw Lambda → static layer")
        return CollapseHWStatic(name="collapse_hw_static")
    return layer

print("Cloning model...")

new_model = keras.models.clone_model(
    model,
    clone_function=replace_layer
)

new_model.set_weights(model.get_weights())

########################################
# Keras 3 SavedModel export
########################################

print("Exporting SavedModel...")
new_model.export("saved_fixed")

########################################
# convert to ONNX
########################################

print("Converting to ONNX...")

spec = (tf.TensorSpec([1,50,212,1], tf.float32, name="image"),)

tf2onnx.convert.from_saved_model(
    "saved_fixed",
    input_signature=spec,
    opset=17,
    output_path="model_fixed.onnx"
)

print("✅ DONE — model_fixed.onnx created")
