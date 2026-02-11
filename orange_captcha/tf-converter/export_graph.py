import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

tf.config.optimizer.set_experimental_options({
    "layout_optimizer": False,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": False,
})

IMG_W = 200
IMG_H = 50
RIGHT_PAD = 12

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x, [s[0], s[1], s[2]*s[3]])

model = keras.models.load_model(
    "ocr_ctc_infer_safe_v9.keras",
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

@tf.function(
    input_signature=[
        tf.TensorSpec([None, IMG_H, IMG_W+RIGHT_PAD, 1], tf.float32)
    ],
    jit_compile=False
)
def serve(x):
    return model(x)

concrete = serve.get_concrete_function()

frozen = convert_variables_to_constants_v2(
    concrete,
    lower_control_flow=False,
    aggressive_inlining=False
)

tf.io.write_graph(
    frozen.graph,
    ".",
    "captcha_tfnet_compatible.pb",
    as_text=False
)

print("Saved captcha_tfnet_compatible.pb")

print("Inputs:", [t.name for t in frozen.inputs])
print("Outputs:", [t.name for t in frozen.outputs])