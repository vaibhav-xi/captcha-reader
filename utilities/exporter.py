import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tf2onnx
import onnxruntime as ort

SRC_MODEL = "captcha_ctc_adapted_v3.keras"
ONNX_OUT  = "captcha_ctc_true.onnx"

class CTCModel(tf.keras.Model):
    def train_step(self,data):
        pass

print("Loading trained model...")
model = tf.keras.models.load_model(
    SRC_MODEL,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("Building concrete forward function...")

@tf.function(input_signature=[
    tf.TensorSpec([None,50,200,1], tf.float32, name="image")
])
def forward(x):
    return model(x, training=False)

print("Tracing...")
concrete = forward.get_concrete_function()

print("Converting to ONNX from concrete function...")

tf2onnx.convert.from_function(
    forward,
    input_signature=[tf.TensorSpec([None,50,200,1], tf.float32, name="image")],
    opset=17,
    output_path=ONNX_OUT
)

print("Validating ONNX runtime load...")
ort.InferenceSession(ONNX_OUT)

print("TRUE ONNX EXPORT COMPLETE")
