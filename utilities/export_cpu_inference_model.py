import tensorflow as tf

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        pass

with tf.device("/CPU:0"):
    model = tf.keras.models.load_model(
        "captcha_ctc_adapted_v3.keras",
        custom_objects={"CTCModel": CTCModel},
        compile=False
    )
    
    dummy = tf.zeros([1, 50, 200, 1], tf.float32)
    _ = model(dummy, training=False)

    model.save("cpu_inference.keras")

print("CPU inference model saved")
