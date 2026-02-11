import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_IN  = "ocr_ctc_infer_safe_v12b.keras"
MODEL_OUT = "ocr_ctc_no_lambda.keras"

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x,[s[0],s[1],s[2]*s[3]])

print("Loading model...")
base = keras.models.load_model(
    MODEL_IN,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Model loaded")

permute_layer = base.get_layer("permute")
after_permute = permute_layer.output

print("Permute output shape:", after_permute.shape)

static = layers.Reshape(
    (53, 768),
    name="reshape_static"
)(after_permute)

print("Inserted static reshape → (53,768)")

x = static
start_copy = False

for layer in base.layers:
    if layer.name == "collapse_hw":
        start_copy = True
        continue
    if not start_copy:
        continue

    if layer.name == "collapse_hw":
        continue

    print("Replaying layer:", layer.name, layer.__class__.__name__)
    x = layer(x)

patched = keras.Model(base.input, x)

copied = 0
for layer in patched.layers:
    try:
        src = base.get_layer(layer.name)
        layer.set_weights(src.get_weights())
        copied += 1
    except:
        pass

print("Weights copied for", copied, "layers")

patched.save(MODEL_OUT)
print("Saved:", MODEL_OUT)

import numpy as np

dummy = np.zeros((1,50,212,1), np.float32)

a = base.predict(dummy, verbose=0)
b = patched.predict(dummy, verbose=0)

print("\nSanity check:")
print("output shapes:", a.shape, b.shape)
print("max abs diff :", np.max(np.abs(a-b)))
print("mean diff    :", np.mean(np.abs(a-b)))
