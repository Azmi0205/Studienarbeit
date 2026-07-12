import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive plot window
import matplotlib.pyplot as plt
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "T_Low_average.npy")
output_image = os.path.join(script_dir, "T_Low_heatmap_counts_fixed.png")


# ----- FIXED COLORBAR SCALE (in raw sensor counts) -----
# Set USE_FIXED_COLORBAR to True to force all heatmaps to share the same scale.
# Set to False to auto-scale based on this image's own min/max instead.
# Conversion: 100 counts = 1 Kelvin, so counts = (Celsius + 273.15) * 100
USE_FIXED_COLORBAR = True
FIXED_VMIN_COUNTS = 31535   # equivalent to 42.2 °C
FIXED_VMAX_COUNTS = 33125   # equivalent to 58.1 °C


# ----- LOAD THE AVERAGE IMAGE (RAW SENSOR COUNTS, NO CONVERSION) -----
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Could not find '{input_file}'")


counts_image = np.load(input_file)
print(f"Loaded average image with shape: {counts_image.shape}")


actual_min = counts_image.min()
actual_max = counts_image.max()
actual_mean = counts_image.mean()


print(f"Actual min counts:  {actual_min:.1f}")
print(f"Actual max counts:  {actual_max:.1f}")
print(f"Actual mean counts: {actual_mean:.1f}")


# ----- RESOLVE COLORBAR BOUNDS -----
if USE_FIXED_COLORBAR:
    vmin = FIXED_VMIN_COUNTS
    vmax = FIXED_VMAX_COUNTS
    if actual_min < vmin or actual_max > vmax:
        print(f"WARNING: image values ({actual_min:.1f} to {actual_max:.1f} counts) "
              f"fall outside the fixed scale ({vmin} to {vmax} counts). "
              "Out-of-range pixels will be clipped visually to the scale's edge colors.")
else:
    vmin = None
    vmax = None
    print("Auto-scaling colorbar based on this image's own min/max.")


# ----- PLOT HEATMAP -----
fig, ax = plt.subplots(figsize=(10, 8))


im = ax.imshow(
    counts_image,
    cmap="inferno",
    vmin=vmin,
    vmax=vmax
)


cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Raw Sensor Counts", fontsize=12)


ax.set_title("Average Thermal Image (Raw Counts)", fontsize=14)
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")


plt.tight_layout()
plt.savefig(output_image, dpi=200)
print(f"\nHeatmap saved to '{output_image}'")


plt.show()  # remove or comment out if running headless