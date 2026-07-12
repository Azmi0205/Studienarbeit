import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive plot window
import matplotlib.pyplot as plt
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_low = os.path.join(script_dir, "T_Low_average.npy")
input_file_high = os.path.join(script_dir, "T_High_noFFC_average.npy")
output_image = os.path.join(script_dir, "SiTF_heatmap.png")
output_npy = os.path.join(script_dir, "SiTF_map.npy")


# ----- KNOWN TARGET TEMPERATURES (in Celsius) -----
T_LOW_C = 40.0
T_HIGH_C = 55.0
DELTA_T = T_HIGH_C - T_LOW_C  # in Kelvin, since ΔT is the same in °C and K


# ----- FIXED COLORBAR SCALE (in counts/K) -----
# Set USE_FIXED_COLORBAR to True to force a consistent scale across runs.
# Set to False to auto-scale based on this map's own min/max instead.
USE_FIXED_COLORBAR = False
FIXED_VMIN = 0.0     # e.g. lower bound in counts/K
FIXED_VMAX = 50.0    # e.g. upper bound in counts/K


# ----- LOAD THE TWO AVERAGED RAW IMAGES -----
if not os.path.isfile(input_file_low):
    raise FileNotFoundError(f"Could not find '{input_file_low}'")
if not os.path.isfile(input_file_high):
    raise FileNotFoundError(f"Could not find '{input_file_high}'")

Y_low = np.load(input_file_low)
Y_high = np.load(input_file_high)

if Y_low.shape != Y_high.shape:
    raise ValueError(
        f"Shape mismatch: T_Low image is {Y_low.shape}, "
        f"T_High image is {Y_high.shape}. Both must match."
    )

print(f"Loaded T_Low image with shape:  {Y_low.shape}")
print(f"Loaded T_High image with shape: {Y_high.shape}")
print(f"Delta T: {DELTA_T} K")


# ----- COMPUTE SiTF PER PIXEL -----
# SiTF_{i,j} = (Y_high - Y_low) / (T_high - T_low)   [counts / K]
SiTF = (Y_high - Y_low) / DELTA_T

sitf_min = SiTF.min()
sitf_max = SiTF.max()
sitf_mean = SiTF.mean()

print(f"\nSiTF min:  {sitf_min:.4f} counts/K")
print(f"SiTF max:  {sitf_max:.4f} counts/K")
print(f"SiTF mean: {sitf_mean:.4f} counts/K")


# ----- SAVE THE SiTF MAP AS .npy -----
np.save(output_npy, SiTF)
print(f"\nSiTF map saved to '{output_npy}'")


# ----- RESOLVE COLORBAR BOUNDS -----
if USE_FIXED_COLORBAR:
    vmin = FIXED_VMIN
    vmax = FIXED_VMAX
    if sitf_min < vmin or sitf_max > vmax:
        print(f"WARNING: SiTF values ({sitf_min:.2f} to {sitf_max:.2f} counts/K) "
              f"fall outside the fixed scale ({vmin} to {vmax} counts/K). "
              "Out-of-range pixels will be clipped visually to the scale's edge colors.")
else:
    vmin = None
    vmax = None
    print("Auto-scaling colorbar based on this map's own min/max.")


# ----- PLOT HEATMAP -----
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    SiTF,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("SiTF (counts/K)", fontsize=12)

ax.set_title(f"Signal Transfer Function (T_low={T_LOW_C}°C, T_high={T_HIGH_C}°C)", fontsize=13)
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")

plt.tight_layout()
plt.savefig(output_image, dpi=200)
print(f"\nHeatmap saved to '{output_image}'")

plt.show()  # remove or comment out if running headless