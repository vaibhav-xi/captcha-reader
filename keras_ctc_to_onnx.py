import tensorflow as tf
import tf2onnx

MODEL_PATH = "orange_captcha/captcha_ctc_adapted_v3.keras"
ONNX_PATH = "captcha_ctc.onnx"

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        pass

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("Loaded model")

spec = (
    tf.TensorSpec((None, 50, 200, 1), tf.float32, name="image"),
)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
)

with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved ONNX →", ONNX_PATH)
