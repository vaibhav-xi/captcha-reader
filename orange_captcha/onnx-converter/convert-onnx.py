import tensorflow as tf
import tf2onnx

KERAS_PATH = "tmp_saved_model.keras"

IMG_H = 50
IMG_W = 212
CH = 1

print("Loading model...")
model = tf.keras.models.load_model(
    KERAS_PATH,
    compile=False
)

input_spec = tf.TensorSpec(
    [None, IMG_H, IMG_W, CH],
    tf.float32,
    name="image"
)

@tf.function(input_signature=[input_spec])
def serving_fn(x):
    return model(x)

print("Converting to ONNX...")

model_proto, _ = tf2onnx.convert.from_function(
    serving_fn,
    input_signature=[input_spec],
    opset=13,
    output_path="model.onnx"
)

print("ONNX saved → model.onnx")
