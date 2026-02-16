import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1) Load data
X = pd.read_csv("X.csv").values
y = pd.read_csv("y.csv", header=None).values.ravel()

# 2) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Reshape for CNN (samples, time_steps, channels)
X_train = X_train.reshape(-1, 128, 1)
X_test = X_test.reshape(-1, 128, 1)

# 4) Build 1D CNN
model = Sequential([
    Conv1D(16, kernel_size=3, activation="relu", input_shape=(128, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 5) Train
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# 6) Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# 7) Save model
output_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Models"
os.makedirs(output_folder, exist_ok=True)

model.save(os.path.join(output_folder, "bearing_fault_model.h5"))
print("Model saved as bearing_fault_model.h5")
