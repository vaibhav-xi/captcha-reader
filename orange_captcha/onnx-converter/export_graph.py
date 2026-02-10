import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

print("TF version:", tf.__version__)

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

@tf.function
def infer(x):
    return feature_model(x)

concrete = infer.get_concrete_function(
    tf.TensorSpec([None,50,212,1], tf.float32)
)

frozen_func = convert_variables_to_constants_v2(concrete)
graph_def = frozen_func.graph.as_graph_def()

tf.io.write_graph(
    graph_def,
    ".",
    "captcha_frozen.pb",
    as_text=False
)

print("Frozen graph saved → captcha_frozen.pb")

print("\nInputs:")
for inp in frozen_func.inputs:
    print(inp.name)

print("\nOutputs:")
for out in frozen_func.outputs:
    print(out.name)
