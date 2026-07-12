import numpy as np
import os


# ----- CONFIGURATION -----
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "NETD_map.npy")
output_filtered_npy = os.path.join(script_dir, "NETD_filtered.npy")
output_stats_file = os.path.join(script_dir, "NETD_filtered_stats.txt")


# ----- OUTLIER FILTERING METHOD -----
# Choose one of: "iqr", "sigma", "percentile"
FILTER_METHOD = "iqr"

# IQR method: values outside [Q1 - IQR_FACTOR*IQR, Q3 + IQR_FACTOR*IQR] are outliers
IQR_FACTOR = 1

# Sigma method: values outside [mean - SIGMA_FACTOR*std, mean + SIGMA_FACTOR*std] are outliers
SIGMA_FACTOR = 3.0

# Percentile method: values outside [LOWER_PCT, UPPER_PCT] are outliers
LOWER_PCT = 1.0
UPPER_PCT = 99.0


# ----- LOAD THE NETD MAP -----
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Could not find '{input_file}'")

NETD_mK = np.load(input_file)
print(f"Loaded NETD map with shape: {NETD_mK.shape}")

netd_flat = NETD_mK.flatten()
valid_values = netd_flat[np.isfinite(netd_flat)]
n_dead = netd_flat.size - valid_values.size

if n_dead > 0:
    print(f"Excluded {n_dead} already-invalid (NaN, e.g. dead) pixel(s) before outlier filtering.")

print(f"Valid pixels before outlier filtering: {valid_values.size}")


# ----- BEFORE-FILTERING STATS -----
raw_mean = valid_values.mean()
raw_median = np.median(valid_values)
raw_std = valid_values.std()

print(f"\nBefore outlier filtering (mK):")
print(f"  mean:   {raw_mean:.4f}")
print(f"  median: {raw_median:.4f}")
print(f"  std:    {raw_std:.4f}")


# ----- DETERMINE OUTLIER BOUNDS -----
if FILTER_METHOD == "iqr":
    Q1 = np.percentile(valid_values, 25)
    Q3 = np.percentile(valid_values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_FACTOR * IQR
    upper_bound = Q3 + IQR_FACTOR * IQR
    print(f"\nMethod: IQR (factor={IQR_FACTOR})")
    print(f"  Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")

elif FILTER_METHOD == "sigma":
    lower_bound = raw_mean - SIGMA_FACTOR * raw_std
    upper_bound = raw_mean + SIGMA_FACTOR * raw_std
    print(f"\nMethod: Sigma clipping (factor={SIGMA_FACTOR})")

elif FILTER_METHOD == "percentile":
    lower_bound = np.percentile(valid_values, LOWER_PCT)
    upper_bound = np.percentile(valid_values, UPPER_PCT)
    print(f"\nMethod: Percentile clipping ({LOWER_PCT}% - {UPPER_PCT}%)")

else:
    raise ValueError(f"Unknown FILTER_METHOD: '{FILTER_METHOD}'")

print(f"  Lower bound: {lower_bound:.4f} mK")
print(f"  Upper bound: {upper_bound:.4f} mK")


# ----- APPLY FILTER -----
outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
n_outliers = np.count_nonzero(outlier_mask)
filtered_values = valid_values[~outlier_mask]

print(f"\nOutliers removed: {n_outliers} ({100 * n_outliers / valid_values.size:.2f}% of valid pixels)")
print(f"Remaining pixels: {filtered_values.size}")


# ----- AFTER-FILTERING STATS -----
filtered_mean = filtered_values.mean()
filtered_median = np.median(filtered_values)
filtered_std = filtered_values.std()
filtered_min = filtered_values.min()
filtered_max = filtered_values.max()

print(f"\nAfter outlier filtering (mK):")
print(f"  mean:   {filtered_mean:.4f}")
print(f"  median: {filtered_median:.4f}")
print(f"  std:    {filtered_std:.4f}")
print(f"  min:    {filtered_min:.4f}")
print(f"  max:    {filtered_max:.4f}")


# ----- SAVE FILTERED MAP (outliers set to NaN, same shape as original) -----
NETD_filtered_map = NETD_mK.copy()
full_outlier_mask = np.isfinite(NETD_filtered_map) & (
    (NETD_filtered_map < lower_bound) | (NETD_filtered_map > upper_bound)
)
NETD_filtered_map[full_outlier_mask] = np.nan

np.save(output_filtered_npy, NETD_filtered_map)
print(f"\nFiltered NETD map (outliers set to NaN) saved to '{output_filtered_npy}'")


# ----- SAVE STATS TO FILE -----
with open(output_stats_file, "w") as f:
    f.write(f"Filter method: {FILTER_METHOD}\n")
    f.write(f"Lower bound: {lower_bound:.4f} mK\n")
    f.write(f"Upper bound: {upper_bound:.4f} mK\n")
    f.write(f"Dead/invalid pixels excluded before filtering: {n_dead}\n")
    f.write(f"Outliers removed: {n_outliers}\n")
    f.write(f"Remaining valid pixels: {filtered_values.size}\n\n")
    f.write(f"Before filtering - mean: {raw_mean:.4f} mK, median: {raw_median:.4f} mK, std: {raw_std:.4f} mK\n")
    f.write(f"After filtering  - mean: {filtered_mean:.4f} mK, median: {filtered_median:.4f} mK, std: {filtered_std:.4f} mK\n")
    f.write(f"After filtering  - min: {filtered_min:.4f} mK, max: {filtered_max:.4f} mK\n")

print(f"Stats saved to '{output_stats_file}'")