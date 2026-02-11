import tensorflow as tf

with open("captcha_tfnet_compatible.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

names = [n.name for n in graph_def.node]

print("\n=== LAST 30 OPS ===")
for n in names[-30:]:
    print(n)

print("\n=== INPUT CANDIDATES ===")
for n in names:
    if "Placeholder" in n or "serving" in n or n == "x":
        print(n)

print("\n=== OUTPUT CANDIDATES ===")
for n in names:
    if any(k in n for k in ["Identity", "PartitionedCall", "Stateful"]):
        print(n)
