import os
import glob
from pathlib import Path
import cv2
import numpy as np

def compute_phash(img_path):
    # Standard 64-bit DCT perceptual hash
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to 32x32
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    # 2D DCT
    dct = cv2.dct(resized.astype(np.float32))
    # Top-left 8x8
    dct_low = dct[0:8, 0:8]
    # Median excluding DC term (0,0)
    med = np.median(dct_low[1:, 1:])
    # 64-bit boolean mask
    hash_bits = (dct_low > med).flatten()
    return hash_bits

# Get directories
train_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\marine_litter\train\images")
fw_test_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\floating_waste\test\images")
flow_test_dir = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\flow_img\test\images")

print(f"Reading train images from: {train_dir}")
train_files = sorted(list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")))
print(f"Total train images: {len(train_files)}")

train_hashes = []
for p in train_files:
    h = compute_phash(p)
    if h is not None:
        train_hashes.append(h)
train_hashes = np.array(train_hashes, dtype=bool) # shape: (N_train, 64)
print(f"Computed {len(train_hashes)} train hashes. Shape: {train_hashes.shape}")

def compare_hashes(test_dir, name):
    test_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    test_hashes = []
    valid_files = []
    for p in test_files:
        h = compute_phash(p)
        if h is not None:
            test_hashes.append(h)
            valid_files.append(p)
    test_hashes = np.array(test_hashes, dtype=bool) # shape: (N_test, 64)
    print(f"\nComputed {len(test_hashes)} {name} test hashes. Shape: {test_hashes.shape}")

    # Compute hamming distance in blocks to save memory:
    # distance = sum(xor)
    matches_le4 = 0
    min_dist_overall = 64
    for i in range(len(test_hashes)):
        th = test_hashes[i:i+1] # (1, 64)
        # XOR with all train hashes
        xor_diff = np.bitwise_xor(train_hashes, th) # (N_train, 64)
        dists = np.sum(xor_diff, axis=1) # (N_train,)
        min_d = np.min(dists)
        if min_d < min_dist_overall:
            min_dist_overall = min_d
        le4 = np.sum(dists <= 4)
        if le4 > 0:
            matches_le4 += le4
            print(f"Found match: {valid_files[i].name} has {le4} matches with dist <= 4 (min dist = {min_d})")

    print(f"[{name}] Total pairs with Hamming distance <= 4: {matches_le4}")
    print(f"[{name}] Global minimum Hamming distance: {min_dist_overall}")
    return matches_le4, min_dist_overall

fw_matches, fw_min = compare_hashes(fw_test_dir, "Floating Waste")
flow_matches, flow_min = compare_hashes(flow_test_dir, "FloW-Img")

