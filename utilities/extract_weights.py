import tensorflow as tf

class CTCModel(tf.keras.Model):
    def train_step(self, data):
        pass

model = tf.keras.models.load_model(
    "captcha_ctc_adapted_v3.keras",
    custom_objects={"CTCModel": CTCModel},
    compile=False
)

model.save_weights("weights.weights.h5")
print("weights saved")