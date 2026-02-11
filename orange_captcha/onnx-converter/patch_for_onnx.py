import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
def collapse_hw(t):
    s = tf.shape(t)
    return tf.reshape(t, [s[0], s[1], s[2]*s[3]])

m = keras.models.load_model(
    "ocr_ctc_infer_safe_v12b.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("Loaded model")

collapse_layer = None
for i, l in enumerate(m.layers):
    if l.name == "collapse_hw":
        collapse_layer = (i, l)
        break

if collapse_layer is None:
    raise RuntimeError("collapse_hw layer not found")

idx = collapse_layer[0]
print("collapse_hw index:", idx)

prev_tensor = m.layers[idx-1].output
shape = prev_tensor.shape

T = shape[1]
H = shape[2]
C = shape[3]

if None in (T,H,C):
    raise RuntimeError("Shape still dynamic — print model.summary()")

F = H * C

print("Static reshape →", (T, F))

x = keras.layers.Reshape((T, F), name="collapse_hw_static")(prev_tensor)

out = x
for l in m.layers[idx+1:]:
    out = l(out)

new_model = keras.Model(m.input, out)

new_model.save("onnx_safe.keras")
print("Saved → onnx_safe.keras")
