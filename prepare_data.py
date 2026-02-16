import scipy.io
import numpy as np
import os
import pandas as pd

# ---------------------- Set correct paths ----------------------
normal_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Data\Normal"
fault_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Data\Bearing Fault"

# ---------------------- Load .mat file safely ----------------------
def load_mat(file_path):
    data = scipy.io.loadmat(file_path)
    # get first key that is not metadata
    key = [k for k in data.keys() if not k.startswith("__")][0]
    signal = data[key]

    # convert to 1D
    signal = np.ravel(signal)
    return signal

# ---------------------- Create 128-sample windows ----------------------
def create_windows(signal, window_size=128, step=128):
    windows = []
    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

X, y = [], []

# ---------------------- Normal data ----------------------
for file in os.listdir(normal_folder):
    if file.endswith(".mat"):
        signal = load_mat(os.path.join(normal_folder, file))
        windows = create_windows(signal)

        if len(windows) > 0:
            X.append(windows)
            y.append(np.zeros(len(windows)))

# ---------------------- Fault data (Drive + Fan) ----------------------
for root, dirs, files in os.walk(fault_folder):
    for file in files:
        if file.endswith(".mat"):
            signal = load_mat(os.path.join(root, file))
            windows = create_windows(signal)

            if len(windows) > 0:
                X.append(windows)
                y.append(np.ones(len(windows)))

# ---------------------- Combine all windows ----------------------
X = np.vstack(X)  # safer than concatenate
y = np.concatenate(y)

# ---------------------- Save to CSV ----------------------
import os
print("Current working directory:", os.getcwd())

pd.DataFrame(X).to_csv("X.csv", index=False)
pd.DataFrame(y).to_csv("y.csv", index=False, header=False)

output_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject"
os.makedirs(output_folder, exist_ok=True)

print("Saving files to:", output_folder)

pd.DataFrame(X).to_csv(os.path.join(output_folder, "X.csv"), index=False)
pd.DataFrame(y).to_csv(os.path.join(output_folder, "y.csv"), index=False, header=False)


