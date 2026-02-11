import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

print("TF version:", tf.__version__)

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

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

@tf.function(
    input_signature=[
        tf.TensorSpec([None, IMG_H, IMG_W+RIGHT_PAD, 1], tf.float32)
    ]
)
def serve(x):
    return model(x)

print("Tracing function...")
concrete = serve.get_concrete_function()

print("Freezing...")
frozen_func = convert_variables_to_constants_v2(concrete)
graph_def = frozen_func.graph.as_graph_def()

tf.io.write_graph(
    graph_def,
    ".",
    "captcha_full_frozen.pb",
    as_text=False
)

print("Frozen graph saved → captcha_full_frozen.pb")

print("\nInputs:")
for inp in frozen_func.inputs:
    print(inp.name, inp.shape)

print("\nOutputs:")
for out in frozen_func.outputs:
    print(out.name, out.shape)
