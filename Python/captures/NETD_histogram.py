import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive plot window
import matplotlib.pyplot as plt
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "NETD_map.npy")
output_image = os.path.join(script_dir, "NETD_histogram.png")


# ----- BIN COUNT -----
# For a 320x256 image (81,920 pixels), the square-root rule gives ~286 bins,
# which is a reasonable, commonly used default for this many samples.
# Set to None to let this be computed automatically via the sqrt rule,
# or override with a fixed integer if you want coarser/finer binning.
NUM_BINS = None


# ----- LOAD THE NETD MAP -----
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Could not find '{input_file}'")

NETD_mK = np.load(input_file)
print(f"Loaded NETD map with shape: {NETD_mK.shape}")

# Flatten and drop invalid (NaN) pixels, e.g. dead pixels excluded earlier
netd_flat = NETD_mK.flatten()
valid_values = netd_flat[np.isfinite(netd_flat)]
n_invalid = netd_flat.size - valid_values.size

if n_invalid > 0:
    print(f"Excluded {n_invalid} invalid (NaN) pixel(s) from the histogram.")

print(f"Valid pixels used: {valid_values.size}")


# ----- RESOLVE NUMBER OF BINS -----
if NUM_BINS is None:
    num_bins = int(np.ceil(np.sqrt(valid_values.size)))
else:
    num_bins = NUM_BINS

print(f"Using {num_bins} bins")


# ----- STATISTICS -----
netd_min = valid_values.min()
netd_max = valid_values.max()
netd_mean = valid_values.mean()
netd_median = np.median(valid_values)
netd_std = valid_values.std()

print(f"\nNETD statistics (mK):")
print(f"  min:    {netd_min:.4f}")
print(f"  max:    {netd_max:.4f}")
print(f"  mean:   {netd_mean:.4f}")
print(f"  median: {netd_median:.4f}")
print(f"  std:    {netd_std:.4f}")


# ----- PLOT HISTOGRAM -----
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(valid_values, bins=num_bins, color="steelblue", edgecolor="black", linewidth=0.3)

ax.axvline(netd_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {netd_mean:.2f} mK")
ax.axvline(netd_median, color="green", linestyle="--", linewidth=1.5, label=f"Median = {netd_median:.2f} mK")

ax.set_xlabel("NETD (mK)", fontsize=12)
ax.set_ylabel("Pixel Count", fontsize=12)
ax.set_title(f"NETD Distribution ({num_bins} bins, n={valid_values.size} pixels)", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_image, dpi=200)
print(f"\nHistogram saved to '{output_image}'")

plt.show()  # remove or comment out if running headless