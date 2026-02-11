import tensorflow as tf

m = tf.saved_model.load("saved_features")

print(list(m.signatures.keys()))
print(m.signatures["serving_default"].structured_input_signature)
print(m.signatures["serving_default"].structured_outputs)