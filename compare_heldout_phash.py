import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def compute_phash(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[0:8, 0:8]
    med = np.median(dct_low[1:, 1:])
    hash_bits = (dct_low > med).flatten()
    return hash_bits

fw_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\floating_waste\test\images")
flow_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\flow_img\test\images")

fw_files = sorted(list(fw_dir.glob("*.jpg")) + list(fw_dir.glob("*.png")))
flow_files = sorted(list(flow_dir.glob("*.jpg")) + list(flow_dir.glob("*.png")))

print(f"Floating Waste test images: {len(fw_files)}")
print(f"FloW-Img test images:       {len(flow_files)}")

fw_hashes = []
valid_fw = []
for f in fw_files:
    h = compute_phash(f)
    if h is not None:
        fw_hashes.append(h)
        valid_fw.append(f)
fw_hashes = np.array(fw_hashes, dtype=bool)

flow_hashes = []
valid_flow = []
for f in flow_files:
    h = compute_phash(f)
    if h is not None:
        flow_hashes.append(h)
        valid_flow.append(f)
flow_hashes = np.array(flow_hashes, dtype=bool)

print(f"Computed FW hashes: {fw_hashes.shape}, FloW hashes: {flow_hashes.shape}")

# Pairwise XOR distance matrix: (306, 599, 64) -> sum -> (306, 599)
# Compute in memory efficiently:
# (N_fw, 1, 64) ^ (1, N_flow, 64)
xor_mat = np.bitwise_xor(fw_hashes[:, np.newaxis, :], flow_hashes[np.newaxis, :, :])
dist_mat = np.sum(xor_mat, axis=2) # shape (306, 599)

total_pairs = dist_mat.size
min_dist = np.min(dist_mat)
mean_dist = np.mean(dist_mat)
median_dist = np.median(dist_mat)

print("\n" + "="*60)
print("PERCEPTUAL HASH (pHash) COMPARISON: Floating Waste vs FloW-Img")
print("="*60)
print(f"Total pairwise comparisons: {total_pairs:,}")
print(f"Global Minimum Hamming Distance: {min_dist} / 64")
print(f"Mean Hamming Distance:           {mean_dist:.2f} / 64")
print(f"Median Hamming Distance:         {median_dist:.1f} / 64")

le0 = np.sum(dist_mat == 0)
le4 = np.sum(dist_mat <= 4)
le6 = np.sum(dist_mat <= 6)
le8 = np.sum(dist_mat <= 8)
le10 = np.sum(dist_mat <= 10)

print(f"Pairs with dist == 0 (exact duplicates): {le0}")
print(f"Pairs with dist <= 4 (near duplicates):  {le4}")
print(f"Pairs with dist <= 6:                   {le6}")
print(f"Pairs with dist <= 8:                   {le8}")
print(f"Pairs with dist <= 10:                  {le10}")

# Find top 10 lowest distance pairs
flat_indices = np.argsort(dist_mat, axis=None)[:10]
print("\nTop 10 Closest Pairs:")
for idx in flat_indices:
    i, j = np.unravel_index(idx, dist_mat.shape)
    print(f"  Dist={dist_mat[i, j]:2d} | FW: {valid_fw[i].name} <---> FloW: {valid_flow[j].name}")

# Specifically check image 400
img400_matches = [i for i, f in enumerate(valid_fw) if "400" in f.name]
if img400_matches:
    i400 = img400_matches[0]
    dists400 = dist_mat[i400, :]
    min400 = np.min(dists400)
    best_flow_idx = np.argmin(dists400)
    print(f"\nSpecific check for Floating Waste image 400 ({valid_fw[i400].name}):")
    print(f"  Min Hamming distance to any FloW-Img image: {min400} / 64")
    print(f"  Closest FloW-Img image: {valid_flow[best_flow_idx].name} (dist={min400})")
    print(f"  Mean distance from img 400 to all FloW images: {np.mean(dists400):.2f}")

# Save CSV of results
df_summary = pd.DataFrame([{
    "comparison": "Floating_Waste_test_vs_FloW_Img_test",
    "total_pairs": total_pairs,
    "min_hamming_dist": int(min_dist),
    "mean_hamming_dist": round(float(mean_dist), 2),
    "median_hamming_dist": float(median_dist),
    "pairs_dist_le_0": int(le0),
    "pairs_dist_le_4": int(le4),
    "pairs_dist_le_6": int(le6),
    "pairs_dist_le_10": int(le10),
    "img400_min_dist": int(min400) if img400_matches else None
}])
csv_path = Path(r"D:\Journal-Project\results_v6\heldout_cross_phash_results.csv")
df_summary.to_csv(csv_path, index=False)
print(f"\nSaved summary to {csv_path}")

