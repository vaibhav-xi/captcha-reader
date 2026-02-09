import onnx
import onnxruntime as ort
import numpy as np

ONNX_PATH = "model.onnx"

print("Loading ONNX...")
model = onnx.load(ONNX_PATH)

print("Running ONNX checker...")
onnx.checker.check_model(model)
print("ONNX structure valid")

ops = set(node.op_type for node in model.graph.node)

print("\nOps used in graph:")
for op in sorted(ops):
    print(" ", op)

BAD = [op for op in ops if "Cudnn" in op or "CuDNN" in op or "Fused" in op]

if BAD:
    print("\nFound GPU/CuDNN fused ops:", BAD)
else:
    print("\nNo CuDNN / fused GPU ops detected")

print("\nInputs:")
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(" ", inp.name, shape)

print("\nOutputs:")
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(" ", out.name, shape)

print("\nCreating ONNX Runtime session...")
sess = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

print("Runtime input:", input_name, input_shape)

test_shape = [d if isinstance(d,int) else 1 for d in input_shape]

print("Test tensor shape:", test_shape)

x = np.random.rand(*test_shape).astype(np.float32)

print("Running inference...")
out = sess.run(None, {input_name: x})

print("Inference OK")
print("Output shapes:")
for o in out:
    print(" ", np.array(o).shape)

print("\nONNX model is SAFE")
