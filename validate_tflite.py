import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="bearing_fault_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get scale & zero point
input_scale, input_zero_point = input_details[0]['quantization']

# Load data
X = pd.read_csv("X.csv").values
y = pd.read_csv("y.csv", header=None).values.ravel()

# Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape
X_test = X_test.reshape(-1, 128, 1).astype(np.float32)

# Quantize using scale and zero_point
X_test_uint8 = np.round(X_test / input_scale + input_zero_point).astype(np.uint8)

correct = 0

for i in range(len(X_test_uint8)):
    sample = X_test_uint8[i:i+1]
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    pred = 1 if output > 0.5 else 0
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test_uint8)
print("TFLite model accuracy:", accuracy)
