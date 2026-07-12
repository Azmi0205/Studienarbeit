import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive plot window
import matplotlib.pyplot as plt
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "HotSpot_Corner_average.npy")
output_mask_image = os.path.join(script_dir, "hotspot_mask.png")


# ----- HOTSPOT DETECTION THRESHOLD -----
# Pixels above (mean + K_SIGMA * std) of the WHOLE image are considered
# candidate hotspot pixels. Increase K_SIGMA if background noise is being
# picked up; decrease it if the hotspot edges are being cut off.
K_SIGMA = 28.0

# Raw counts -> Celsius conversion (same convention as your other scripts)
RAW_IS_KELVIN_X100 = True  # set False if your file is already in real units


# ----- LOAD THE AVERAGED IMAGE -----
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Could not find '{input_file}'")

raw_image = np.load(input_file).astype(float)
print(f"Loaded image with shape: {raw_image.shape}")

if RAW_IS_KELVIN_X100:
    temp_image = (raw_image / 100.0) - 273.15  # Celsius
    unit_label = "°C"
else:
    temp_image = raw_image
    unit_label = "raw units"


# ----- ESTIMATE BACKGROUND STATISTICS -----
bg_mean = raw_image.mean()
bg_std = raw_image.std()
threshold = bg_mean + K_SIGMA * bg_std

print(f"\nBackground mean: {bg_mean:.4f}")
print(f"Background std:  {bg_std:.4f}")
print(f"Threshold:        {threshold:.4f} (mean + {K_SIGMA}*std)")


# ----- THRESHOLD AND ISOLATE THE HOTSPOT -----
binary_mask = raw_image > threshold

if not binary_mask.any():
    raise ValueError(
        "No pixels exceeded the threshold. Try lowering K_SIGMA."
    )

# Label connected regions so background noise pixels don't get mixed in
# with the actual hotspot blob
labeled_array, num_features = ndimage.label(binary_mask)
print(f"\nConnected candidate region(s) found: {num_features}")

# Assume the hotspot is the region containing the single brightest pixel
# (robust even if noise creates a few small stray candidate pixels elsewhere)
max_pixel_index = np.unravel_index(np.argmax(raw_image), raw_image.shape)
hotspot_label = labeled_array[max_pixel_index]

if hotspot_label == 0:
    raise ValueError("The brightest pixel was not classified as part of any region.")

hotspot_mask = labeled_array == hotspot_label
hotspot_pixel_count = hotspot_mask.sum()

print(f"Hotspot selected: label {hotspot_label}, {hotspot_pixel_count} pixel(s)")

if num_features > 1:
    print("NOTE: multiple candidate regions were found; only the region "
          "containing the brightest pixel was kept as the hotspot. "
          "Other regions were treated as noise/background artifacts.")


# ----- COMPUTE HOTSPOT STATISTICS -----
hotspot_values_temp = temp_image[hotspot_mask]

hotspot_mean = hotspot_values_temp.mean()
hotspot_min = hotspot_values_temp.min()
hotspot_max = hotspot_values_temp.max()
hotspot_std = hotspot_values_temp.std()

print(f"\nHotspot statistics ({unit_label}):")
print(f"  mean: {hotspot_mean:.4f}")
print(f"  min:  {hotspot_min:.4f}")
print(f"  max:  {hotspot_max:.4f}")
print(f"  std:  {hotspot_std:.4f}")
print(f"  pixel count: {hotspot_pixel_count}")


# ----- VISUALIZE THE DETECTED MASK OVER THE IMAGE -----
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(temp_image, cmap="inferno")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(f"Temperature ({unit_label})", fontsize=12)

# Overlay hotspot boundary as a contour
ax.contour(hotspot_mask, colors="cyan", linewidths=1.5, levels=[0.5])

ax.set_title(f"Detected Hotspot (mean={hotspot_mean:.2f} {unit_label}, "
             f"n={hotspot_pixel_count} px)", fontsize=13)
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")

plt.tight_layout()
plt.savefig(output_mask_image, dpi=200)
print(f"\nHotspot overlay saved to '{output_mask_image}'")

plt.show()  # remove or comment out if running headless