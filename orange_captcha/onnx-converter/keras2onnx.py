import tensorflow as tf
import tf2onnx

KERAS_PATH = "ocr_ctc_infer.keras"

print("Loading keras model...")
model = tf.keras.models.load_model(
    KERAS_PATH,
    compile=False
)

print("Model input:", model.inputs)
print("Model output:", model.outputs)

spec = (
    tf.TensorSpec(
        model.inputs[0].shape,
        tf.float32,
        name="image"
    ),
)

print("Converting (from_keras)...")

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="ocr_safe.onnx"
)

print("✅ Saved → ocr_safe.onnx")
