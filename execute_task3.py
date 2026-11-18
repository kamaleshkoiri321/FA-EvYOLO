"""
execute_task3.py - Task 3: Perceptual Hash Duplicate & Overlap Check
Algorithm: Standard 64-bit DCT perceptual hash (pHash)
- Grayscale, resize 32x32 INTER_AREA
- 2D DCT, top-left 8x8 excluding DC term (0, 0)
- Compare to median -> 64-bit boolean mask
- Hamming distance = bitwise XOR sum

Evaluates:
(a) Floating Waste test vs FloW-Img test (183,294 pairs)
(b) Within the Floating Waste test set (46,665 distinct pairs, i < j)
Outputs CSVs to results_v6/final_audit/
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

OUT_DIR = Path(r"D:\Journal-Project\results_v6\final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_phash(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[0:8, 0:8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

fw_dir   = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\floating_waste\test\images")
flow_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\flow_img\test\images")

fw_files   = sorted(list(fw_dir.glob("*.jpg")) + list(fw_dir.glob("*.png")))
flow_files = sorted(list(flow_dir.glob("*.jpg")) + list(flow_dir.glob("*.png")))

print(f"Floating Waste test images: {len(fw_files)}")
print(f"FloW-Img test images:       {len(flow_files)}")

# 1. Compute hashes
fw_hashes, valid_fw = [], []
for f in fw_files:
    h = compute_phash(f)
    if h is not None:
        fw_hashes.append(h)
        valid_fw.append(f)
fw_hashes = np.array(fw_hashes, dtype=bool)

flow_hashes, valid_flow = [], []
for f in flow_files:
    h = compute_phash(f)
    if h is not None:
        flow_hashes.append(h)
        valid_flow.append(f)
flow_hashes = np.array(flow_hashes, dtype=bool)

# ---------------------------------------------------------------------------
# (a) Floating Waste test vs FloW-Img test
# ---------------------------------------------------------------------------
xor_ab = np.bitwise_xor(fw_hashes[:, np.newaxis, :], flow_hashes[np.newaxis, :, :])
dist_ab = np.sum(xor_ab, axis=2) # (306, 599)

n_pairs_ab = dist_ab.size
min_dist_ab = int(np.min(dist_ab))
mean_dist_ab = float(np.mean(dist_ab))
median_dist_ab = float(np.median(dist_ab))
std_dist_ab = float(np.std(dist_ab))

le0_ab = int(np.sum(dist_ab == 0))
le4_ab = int(np.sum(dist_ab <= 4))
le6_ab = int(np.sum(dist_ab <= 6))
le10_ab = int(np.sum(dist_ab <= 10))

# ---------------------------------------------------------------------------
# (b) Within Floating Waste test set (unique pairs i < j)
# ---------------------------------------------------------------------------
xor_within = np.bitwise_xor(fw_hashes[:, np.newaxis, :], fw_hashes[np.newaxis, :, :])
dist_within = np.sum(xor_within, axis=2) # (306, 306)

# Extract upper triangle without diagonal
triu_indices = np.triu_indices(len(valid_fw), k=1)
dist_within_pairs = dist_within[triu_indices]

n_pairs_within = dist_within_pairs.size
min_dist_within = int(np.min(dist_within_pairs))
mean_dist_within = float(np.mean(dist_within_pairs))
median_dist_within = float(np.median(dist_within_pairs))
std_dist_within = float(np.std(dist_within_pairs))

le0_within = int(np.sum(dist_within_pairs == 0))
le4_within = int(np.sum(dist_within_pairs <= 4))
le6_within = int(np.sum(dist_within_pairs <= 6))
le10_within = int(np.sum(dist_within_pairs <= 10))

# Collect near-duplicate pairs within Floating Waste (dist <= 4)
near_dup_records = []
for idx in range(len(dist_within_pairs)):
    d = int(dist_within_pairs[idx])
    if d <= 4:
        i = int(triu_indices[0][idx])
        j = int(triu_indices[1][idx])
        near_dup_records.append({
            "image_1": valid_fw[i].name,
            "image_2": valid_fw[j].name,
            "hamming_distance": d
        })

df_near_dups = pd.DataFrame(near_dup_records)
near_dups_csv = OUT_DIR / "task3_floating_waste_near_duplicate_pairs.csv"
df_near_dups.to_csv(near_dups_csv, index=False)

# Summary table
summary = [
    {
        "comparison": "(a) Floating Waste test vs FloW-Img test",
        "total_pairs": n_pairs_ab,
        "min_hamming_distance": min_dist_ab,
        "pairs_le_4": le4_ab,
        "pairs_le_0_exact": le0_ab,
        "pairs_le_6": le6_ab,
        "pairs_le_10": le10_ab,
        "mean_distance": mean_dist_ab,
        "median_distance": median_dist_ab,
        "std_distance": std_dist_ab,
        "hash_type": "64-bit DCT perceptual hash (pHash)"
    },
    {
        "comparison": "(b) Within Floating Waste test set",
        "total_pairs": n_pairs_within,
        "min_hamming_distance": min_dist_within,
        "pairs_le_4": le4_within,
        "pairs_le_0_exact": le0_within,
        "pairs_le_6": le6_within,
        "pairs_le_10": le10_within,
        "mean_distance": mean_dist_within,
        "median_distance": median_dist_within,
        "std_distance": std_dist_within,
        "hash_type": "64-bit DCT perceptual hash (pHash)"
    }
]

df_summary = pd.DataFrame(summary)
summary_csv = OUT_DIR / "task3_duplicate_check.csv"
df_summary.to_csv(summary_csv, index=False)

print("\n" + "="*80)
print("TASK 3 DUPLICATE & OVERLAP CHECK RESULTS (Unrounded)")
print("="*80)
for _, r in df_summary.iterrows():
    print(f"\n{r['comparison']}")
    print(f"  Total pairs:               {r['total_pairs']:,}")
    print(f"  Min Hamming distance:      {r['min_hamming_distance']} / 64")
    print(f"  Pairs with distance <= 4:  {r['pairs_le_4']}")
    print(f"  Pairs with distance == 0:  {r['pairs_le_0_exact']}")
    print(f"  Pairs with distance <= 10: {r['pairs_le_10']}")
    print(f"  Mean +- std distance:      {r['mean_distance']:.6f} +- {r['std_distance']:.6f}")
    print(f"  Median distance:           {r['median_distance']:.1f}")

if len(near_dup_records) > 0:
    print(f"\nFound {len(near_dup_records)} near-duplicate pairs (dist <= 4) within Floating Waste test set:")
    for r in near_dup_records[:10]:
        print(f"  Dist={r['hamming_distance']} | {r['image_1']} <---> {r['image_2']}")
    if len(near_dup_records) > 10:
        print(f"  ... and {len(near_dup_records) - 10} more pairs (saved to {near_dups_csv})")
else:
    print("\nNo near-duplicate pairs (dist <= 4) found within Floating Waste test set.")

