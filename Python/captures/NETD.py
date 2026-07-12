import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive plot window
import matplotlib.pyplot as plt
import glob
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_frames_folder = os.path.join(script_dir, "T_Low")   # raw 50 frames used for noise calc
sitf_file = os.path.join(script_dir, "SiTF_map.npy")     # from the previous SiTF script

output_std_npy = os.path.join(script_dir, "T_Low_std.npy")
output_std_image = os.path.join(script_dir, "T_Low_std_heatmap.png")
output_netd_npy = os.path.join(script_dir, "NETD_map.npy")
output_netd_image = os.path.join(script_dir, "NETD_heatmap.png")


# ----- FIXED COLORBAR SCALES -----
# Set to True to force a consistent scale across runs, False to auto-scale.
USE_FIXED_COLORBAR_STD = False
FIXED_VMIN_STD = 0.0     # counts
FIXED_VMAX_STD = 20.0    # counts

USE_FIXED_COLORBAR_NETD = False
FIXED_VMIN_NETD = 0.0    # mK
FIXED_VMAX_NETD = 100.0  # mK


# ----- LOAD RAW FRAMES AND COMPUTE STANDARD DEVIATION -----
file_paths = sorted(glob.glob(os.path.join(raw_frames_folder, "*.npy")))

if len(file_paths) == 0:
    raise FileNotFoundError(f"No .npy files found in '{raw_frames_folder}'")

print(f"Found {len(file_paths)} raw frames in '{raw_frames_folder}'")

images = [np.load(p) for p in file_paths]
stack = np.stack(images, axis=0)

# Per-pixel temporal standard deviation across all frames (in counts)
std_image = np.std(stack, axis=0)

std_min = std_image.min()
std_max = std_image.max()
std_mean = std_image.mean()

print(f"\nStandard deviation (counts):")
print(f"  min:  {std_min:.4f}")
print(f"  max:  {std_max:.4f}")
print(f"  mean: {std_mean:.4f}")

np.save(output_std_npy, std_image)
print(f"Standard deviation map saved to '{output_std_npy}'")


# ----- LOAD SiTF MAP -----
if not os.path.isfile(sitf_file):
    raise FileNotFoundError(f"Could not find '{sitf_file}'")

SiTF = np.load(sitf_file)

if SiTF.shape != std_image.shape:
    raise ValueError(
        f"Shape mismatch: std map is {std_image.shape}, "
        f"SiTF map is {SiTF.shape}. Both must match."
    )


# ----- COMPUTE NETD PER PIXEL -----
# NETD_{i,j} = sigma_{i,j} / SiTF_{i,j}   [counts] / [counts/K] = [K]
# Convert to mK for typical thermal imaging reporting convention.
with np.errstate(divide='ignore', invalid='ignore'):
    NETD_K = std_image / SiTF
NETD_mK = NETD_K * 1000.0

# Guard against divide-by-zero or negative SiTF pixels (dead/bad pixels)
invalid_mask = ~np.isfinite(NETD_mK) | (SiTF <= 0)
n_invalid = np.count_nonzero(invalid_mask)
if n_invalid > 0:
    print(f"\nWARNING: {n_invalid} pixel(s) have invalid NETD (SiTF <= 0 or divide-by-zero). "
          "These are set to NaN and excluded from statistics.")
    NETD_mK[invalid_mask] = np.nan

netd_min = np.nanmin(NETD_mK)
netd_max = np.nanmax(NETD_mK)
netd_mean = np.nanmean(NETD_mK)

print(f"\nNETD (mK):")
print(f"  min:  {netd_min:.4f}")
print(f"  max:  {netd_max:.4f}")
print(f"  mean: {netd_mean:.4f}")

np.save(output_netd_npy, NETD_mK)
print(f"NETD map saved to '{output_netd_npy}'")


# ----- HELPER TO RESOLVE VMIN/VMAX -----
def resolve_bounds(use_fixed, fixed_vmin, fixed_vmax, actual_min, actual_max, label):
    if use_fixed:
        if actual_min < fixed_vmin or actual_max > fixed_vmax:
            print(f"WARNING: {label} values ({actual_min:.2f} to {actual_max:.2f}) "
                  f"fall outside the fixed scale ({fixed_vmin} to {fixed_vmax}). "
                  "Out-of-range pixels will be clipped visually to the scale's edge colors.")
        return fixed_vmin, fixed_vmax
    else:
        print(f"Auto-scaling {label} colorbar based on this map's own min/max.")
        return None, None


# ----- PLOT STANDARD DEVIATION HEATMAP -----
vmin_std, vmax_std = resolve_bounds(
    USE_FIXED_COLORBAR_STD, FIXED_VMIN_STD, FIXED_VMAX_STD, std_min, std_max, "std dev"
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(std_image, cmap="viridis", vmin=vmin_std, vmax=vmax_std)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Standard Deviation (counts)", fontsize=12)
ax.set_title("Temporal Noise (Standard Deviation) at T_Low", fontsize=13)
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
plt.tight_layout()
plt.savefig(output_std_image, dpi=200)
print(f"\nStd dev heatmap saved to '{output_std_image}'")
plt.close(fig)


# ----- PLOT NETD HEATMAP -----
vmin_netd, vmax_netd = resolve_bounds(
    USE_FIXED_COLORBAR_NETD, FIXED_VMIN_NETD, FIXED_VMAX_NETD, netd_min, netd_max, "NETD"
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(NETD_mK, cmap="inferno", vmin=vmin_netd, vmax=vmax_netd)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("NETD (mK)", fontsize=12)
ax.set_title("Noise Equivalent Temperature Difference (NETD)", fontsize=13)
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
plt.tight_layout()
plt.savefig(output_netd_image, dpi=200)
print(f"\nNETD heatmap saved to '{output_netd_image}'")
plt.close(fig)


plt.show()  # remove or comment out if running headless