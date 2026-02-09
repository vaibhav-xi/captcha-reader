import tensorflow as tf

print("Loading keras v3 model...")
m = tf.keras.models.load_model("ocr_ctc_infer.keras", compile=False)

print("Saving legacy H5 model...")
m.save("old_model_legacy.h5")

print("Done → old_model_legacy.h5")
