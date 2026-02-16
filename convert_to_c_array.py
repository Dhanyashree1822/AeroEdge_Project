import pathlib
import os

# Your project folder
output_folder = r"C:\Users\Rachuri Dhanyashree\OneDrive\Desktop\AeroEdgeProject\Models"
os.makedirs(output_folder, exist_ok=True)

tflite_model_path = os.path.join(output_folder, "bearing_fault_model_quant.tflite")
output_cc_path = os.path.join(output_folder, "model_data.cc")
output_h_path = os.path.join(output_folder, "model_data.h")

print("Saving files to:", output_folder)

tflite_model = pathlib.Path(tflite_model_path).read_bytes()

with open(output_cc_path, "w") as f:
    f.write("#include \"model_data.h\"\n\n")
    f.write("const unsigned char model_data[] = {")
    f.write(",".join(str(b) for b in tflite_model))
    f.write("};\n")
    f.write("const unsigned int model_data_len = " + str(len(tflite_model)) + ";\n")

with open(output_h_path, "w") as f:
    f.write("#ifndef MODEL_DATA_H\n")
    f.write("#define MODEL_DATA_H\n\n")
    f.write("extern const unsigned char model_data[];\n")
    f.write("extern const unsigned int model_data_len;\n\n")
    f.write("#endif\n")

print("Files created:")
print(output_cc_path)
print(output_h_path)
