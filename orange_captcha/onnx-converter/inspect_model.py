import tensorflow as tf
from tensorflow import keras

MODEL = "ocr_ctc_infer_safe_v12b.keras"

@tf.keras.utils.register_keras_serializable()
def collapse_hw(x):
    s = tf.shape(x)
    return tf.reshape(x,[s[0],s[1],s[2]*s[3]])

model = keras.models.load_model(
    MODEL,
    compile=False,
    custom_objects={"collapse_hw": collapse_hw}
)

print("\n================ MODEL SUMMARY ================\n")
model.summary(line_length=140)

print("\n================ LAYER DETAILS ================\n")

for i,l in enumerate(model.layers):
    try:
        out = l.output_shape
    except:
        out = "?"
    print(f"{i:3d} | {l.name:30s} | {l.__class__.__name__:20s} | {out}")

print("\n================ HEAD PATH ================\n")

dense = model.layers[-1]
print("Final layer:", dense.name, dense.__class__.__name__)
print("Dense input tensor:", dense.input)
print("Dense input shape :", dense.input.shape)
print("Dense kernel shape:", dense.get_weights()[0].shape)

print("\n================ TRAINABLE COUNT ================\n")

total = 0
for v in model.trainable_variables:
    print(v.name, v.shape)
    total += tf.size(v).numpy()

print("\nTrainable parameter count:", total)

print("\n================ GRAPH WALK BACK FROM HEAD ================\n")

t = model.layers[-1].input
seen = set()

for _ in range(15):
    kh = t._keras_history
    layer = kh[0] if isinstance(kh, tuple) else kh.layer

    if layer.name in seen:
        break
    seen.add(layer.name)

    print("←", layer.name, layer.__class__.__name__)

    inp = layer.input
    if inp is None:
        break

    t = layer.input if not isinstance(layer.input, list) else layer.input[0]