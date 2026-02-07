import tensorflow as tf
import tf2onnx

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        pass

model = tf.keras.models.load_model(
    "captcha_ctc_adapted_v3.keras",
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

print("Loaded keras model")

model.trainable = False

input_signature = [
    tf.TensorSpec([None, 50, 200, 1], tf.float32, name="image")
]

@tf.function(input_signature=input_signature)
def infer(x):
    return model(x, training=False)

onnx_model, _ = tf2onnx.convert.from_function(
    infer,
    input_signature=input_signature,
    opset=17
)

with open("captcha_ctc.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX export complete")
