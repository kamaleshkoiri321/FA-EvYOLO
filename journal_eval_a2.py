"""
Task A2: Paired bootstrap confidence intervals using the validated per-image stats from A1.
YOLO11n seed 42 vs FA-EvYOLO run B seed 42.
10,000 resamples, seed 0.
FloW-Img resampled by source scene (filename before "_jpg").
Reports AP50 and AP50:95 on Floating Waste and FloW-Img.
"""

import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Journal-Project")
from ultralytics.utils.metrics import ap_per_class

OUT_DIR = Path(r"D:\Journal-Project\results_v6\journal_eval")
CACHE_DIR = OUT_DIR / "a1_cache"

def load_per_image(model_name, ds_name):
    cache_path = CACHE_DIR / f"{model_name}_{ds_name}.pkl"
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["per_image"]

def compute_ap_from_stats(stats_list):
    if not stats_list:
        return 0.0, 0.0
    tp_all = np.concatenate([s["tp"] for s in stats_list], axis=0) if sum(len(s["tp"]) for s in stats_list) > 0 else np.zeros((0, 10), dtype=bool)
    conf_all = np.concatenate([s["conf"] for s in stats_list], axis=0) if sum(len(s["conf"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate([s["pred_cls"] for s in stats_list], axis=0) if sum(len(s["pred_cls"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)
    target_cls_all = np.concatenate([s["target_cls"] for s in stats_list], axis=0) if sum(len(s["target_cls"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)
    if len(target_cls_all) == 0:
        return 0.0, 0.0
    ret = ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all,
                       plot=False, save_dir=Path("."), names={0: "item"})
    ap = ret[5]
    ap50 = float(ap[0, 0]) if len(ap) else 0.0
    ap50_95 = float(ap[0].mean()) if len(ap) else 0.0
    return ap50, ap50_95


def bootstrap_paired(base_per_image, fa_per_image, resample_units, n_boot=10000, seed=0):
    """
    Paired bootstrap: resample by units (images or scenes), compute AP difference.
    Returns observed APs and bootstrap CI.
    """
    rng = np.random.RandomState(seed)
    n_units = len(resample_units)

    # Map unit -> list of filenames
    unit_to_files = {}
    all_files = set(base_per_image.keys()) & set(fa_per_image.keys())
    for fn in all_files:
        for unit in resample_units:
            if unit == fn or fn.startswith(unit):
                unit_to_files.setdefault(unit, []).append(fn)
                break

    unit_keys = sorted(unit_to_files.keys())
    n_units = len(unit_keys)

    # Observed
    base_stats = [base_per_image[fn] for fn in sorted(all_files)]
    fa_stats = [fa_per_image[fn] for fn in sorted(all_files)]
    base_ap50, base_ap50_95 = compute_ap_from_stats(base_stats)
    fa_ap50, fa_ap50_95 = compute_ap_from_stats(fa_stats)

    diff_ap50_obs = fa_ap50 - base_ap50
    diff_ap50_95_obs = fa_ap50_95 - base_ap50_95

    boot_diffs_50 = []
    boot_diffs_95 = []

    for b in range(n_boot):
        if b % 1000 == 0:
            print(f"  Bootstrap iteration {b}/{n_boot}")
        # Resample units with replacement
        sampled_units = rng.choice(unit_keys, size=n_units, replace=True)

        base_boot = []
        fa_boot = []
        for unit in sampled_units:
            for fn in unit_to_files[unit]:
                base_boot.append(base_per_image[fn])
                fa_boot.append(fa_per_image[fn])

        b_ap50, b_ap50_95 = compute_ap_from_stats(base_boot)
        f_ap50, f_ap50_95 = compute_ap_from_stats(fa_boot)
        boot_diffs_50.append(f_ap50 - b_ap50)
        boot_diffs_95.append(f_ap50_95 - b_ap50_95)

    boot_diffs_50 = np.array(boot_diffs_50)
    boot_diffs_95 = np.array(boot_diffs_95)

    ci_lower_50 = float(np.percentile(boot_diffs_50, 2.5))
    ci_upper_50 = float(np.percentile(boot_diffs_50, 97.5))
    ci_lower_95 = float(np.percentile(boot_diffs_95, 2.5))
    ci_upper_95 = float(np.percentile(boot_diffs_95, 97.5))

    return {
        "base_ap50": base_ap50, "fa_ap50": fa_ap50,
        "base_ap50_95": base_ap50_95, "fa_ap50_95": fa_ap50_95,
        "diff_ap50": diff_ap50_obs, "diff_ap50_95": diff_ap50_95_obs,
        "ci_lower_ap50": ci_lower_50, "ci_upper_ap50": ci_upper_50,
        "ci_lower_ap50_95": ci_lower_95, "ci_upper_ap50_95": ci_upper_95,
        "n_units": n_units, "n_images": len(all_files),
        "boot_diffs_50": boot_diffs_50, "boot_diffs_95": boot_diffs_95,
    }


def main():
    print("=" * 80)
    print("TASK A2: Paired bootstrap CIs (YOLO11n s42 vs FA-EvYOLO RunB s42)")
    print("=" * 80)

    rows = []

    # Floating Waste: resample by individual image
    print("\n--- Floating Waste (resample by individual image) ---")
    base_fw = load_per_image("YOLO11n_seed42", "floating_waste")
    fa_fw = load_per_image("FA_EvYOLO_RunB_seed42", "floating_waste")
    units_fw = sorted(base_fw.keys())

    r_fw = bootstrap_paired(base_fw, fa_fw, units_fw, n_boot=10000, seed=0)
    rows.append({
        "dataset": "floating_waste",
        "resampling_unit": "Individual test image",
        "n_units": r_fw["n_units"],
        "total_images": r_fw["n_images"],
        "observed_base_ap50": r_fw["base_ap50"],
        "observed_fa_ap50": r_fw["fa_ap50"],
        "observed_diff_ap50": r_fw["diff_ap50"],
        "ci95_lower_diff_ap50": r_fw["ci_lower_ap50"],
        "ci95_upper_diff_ap50": r_fw["ci_upper_ap50"],
        "observed_base_ap50_95": r_fw["base_ap50_95"],
        "observed_fa_ap50_95": r_fw["fa_ap50_95"],
        "observed_diff_ap50_95": r_fw["diff_ap50_95"],
        "ci95_lower_diff_ap50_95": r_fw["ci_lower_ap50_95"],
        "ci95_upper_diff_ap50_95": r_fw["ci_upper_ap50_95"],
    })
    print(f"  AP50: {r_fw['diff_ap50']:+.6f} [{r_fw['ci_lower_ap50']:+.6f}, {r_fw['ci_upper_ap50']:+.6f}]")
    print(f"  AP50:95: {r_fw['diff_ap50_95']:+.6f} [{r_fw['ci_lower_ap50_95']:+.6f}, {r_fw['ci_upper_ap50_95']:+.6f}]")

    # FloW-Img: resample by source scene (filename before "_jpg")
    print("\n--- FloW-Img (resample by source scene) ---")
    base_fi = load_per_image("YOLO11n_seed42", "flow_img")
    fa_fi = load_per_image("FA_EvYOLO_RunB_seed42", "flow_img")

    # Extract scene keys
    scenes = set()
    for fn in base_fi.keys():
        scene = fn.split("_jpg")[0]
        scenes.add(scene)
    scenes = sorted(scenes)
    print(f"  Found {len(scenes)} unique scenes")

    # Build scene -> files mapping
    scene_to_files = {}
    for fn in sorted(base_fi.keys()):
        scene = fn.split("_jpg")[0]
        scene_to_files.setdefault(scene, []).append(fn)

    r_fi = bootstrap_paired(base_fi, fa_fi, scenes, n_boot=10000, seed=0)
    rows.append({
        "dataset": "flow_img",
        "resampling_unit": "Cluster (source image prefix before '_jpg')",
        "n_units": r_fi["n_units"],
        "total_images": r_fi["n_images"],
        "observed_base_ap50": r_fi["base_ap50"],
        "observed_fa_ap50": r_fi["fa_ap50"],
        "observed_diff_ap50": r_fi["diff_ap50"],
        "ci95_lower_diff_ap50": r_fi["ci_lower_ap50"],
        "ci95_upper_diff_ap50": r_fi["ci_upper_ap50"],
        "observed_base_ap50_95": r_fi["base_ap50_95"],
        "observed_fa_ap50_95": r_fi["fa_ap50_95"],
        "observed_diff_ap50_95": r_fi["diff_ap50_95"],
        "ci95_lower_diff_ap50_95": r_fi["ci_lower_ap50_95"],
        "ci95_upper_diff_ap50_95": r_fi["ci_upper_ap50_95"],
    })
    print(f"  AP50: {r_fi['diff_ap50']:+.6f} [{r_fi['ci_lower_ap50']:+.6f}, {r_fi['ci_upper_ap50']:+.6f}]")
    print(f"  AP50:95: {r_fi['diff_ap50_95']:+.6f} [{r_fi['ci_lower_ap50_95']:+.6f}, {r_fi['ci_upper_ap50_95']:+.6f}]")

    # Save raw bootstrap diffs
    raw = pd.DataFrame({
        "fw_diff_ap50": r_fw["boot_diffs_50"],
        "fw_diff_ap50_95": r_fw["boot_diffs_95"],
        "fi_diff_ap50": r_fi["boot_diffs_50"],
        "fi_diff_ap50_95": r_fi["boot_diffs_95"],
    })
    raw.to_csv(OUT_DIR / "a_bootstrap_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "a_bootstrap.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'a_bootstrap.csv'}")


if __name__ == "__main__":
    main()
