import onnxruntime as ort

ort.InferenceSession("captcha_ctc.onnx")
print("ONNX model valid")