import tensorflow as tf
import numpy as np
import pandas as pd

# Load model
model = tf.keras.models.load_model(
    r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Models\bearing_fault_model.h5"
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ------------------ INT8 Quantization ------------------
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use a small subset for calibration
X = pd.read_csv("X.csv").values
X = X.reshape(-1, 128, 1).astype(np.float32)

def representative_data_gen():
    for i in range(100):
        yield [X[i:i+1]]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save
with open("bearing_fault_model_quant.tflite", "wb") as f:
    f.write(tflite_model)
import os

output_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Models"
os.makedirs(output_folder, exist_ok=True)

with open(os.path.join(output_folder, "bearing_fault_model_quant.tflite"), "wb") as f:
    f.write(tflite_model)

print("Saved quantized TFLite model at:", output_folder)

print("Saved quantized TFLite model!")
