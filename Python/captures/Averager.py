import numpy as np
import glob
import os

# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "T_High_noFFC")  # folder containing .npy files to average
output_file = os.path.join(script_dir, "T_High_noFFC_average.npy")
stats_file = os.path.join(script_dir, "T_High_noFFC_stats.txt")

# ----- DEBUG INFO -----
print(f"Script directory: {script_dir}")
print(f"Looking for .npy files in: {input_folder}")
print(f"Folder exists: {os.path.isdir(input_folder)}")

# ----- LOAD ALL .npy FILES (RAW SENSOR COUNTS) -----
file_paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))

if len(file_paths) == 0:
    if os.path.isdir(input_folder):
        print("Folder exists but no .npy files matched. Contents of folder:")
        print(os.listdir(input_folder))
    raise FileNotFoundError(f"No .npy files found in '{input_folder}'")

print(f"Found {len(file_paths)} .npy files.")

images = []
for path in file_paths:
    arr = np.load(path)          # load without any dtype conversion
    images.append(arr)

# Preserve the sensor's native dtype (e.g. uint16 for raw counts)
raw_dtype = images[0].dtype
stack = np.stack(images, axis=0)

# ----- COMPUTE PIXEL-WISE AVERAGE COUNTS ACROSS ALL FILES -----
# Average in float internally for precision, then round back to integer counts
average_counts = np.round(np.mean(stack, axis=0)).astype(raw_dtype)

# ----- SAVE THE AVERAGE COUNT IMAGE -----
np.save(output_file, average_counts)
print(f"Average raw-count image saved to '{output_file}'")

# ----- STATISTICS ON THE AVERAGED COUNT IMAGE -----
highest_avg_pixel = int(np.max(average_counts))
lowest_avg_pixel = int(np.min(average_counts))
overall_average = np.mean(average_counts)

print("\n--- Raw Sensor Count Statistics ---")
print(f"Highest average pixel count: {highest_avg_pixel}")
print(f"Lowest average pixel count:  {lowest_avg_pixel}")
print(f"Average of all average pixel counts: {overall_average:.4f}")

# ----- SAVE STATS TO A TEXT FILE -----
with open(stats_file, "w") as f:
    f.write(f"Number of images processed: {len(file_paths)}\n")
    f.write(f"Image shape: {average_counts.shape}\n")
    f.write(f"Data type: {average_counts.dtype}\n")
    f.write(f"Highest average pixel count: {highest_avg_pixel}\n")
    f.write(f"Lowest average pixel count: {lowest_avg_pixel}\n")
    f.write(f"Average of all average pixel counts: {overall_average:.4f}\n")

print(f"\nStats also saved to '{stats_file}'")