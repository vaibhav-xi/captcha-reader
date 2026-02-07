import onnx
m = onnx.load("captcha_ctc.onnx")
print(m.graph.input[0])