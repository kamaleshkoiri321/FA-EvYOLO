"""
Task A3: Redo brightness-tertile split using validated per-image stats from A1.
Uses the SAME tertile assignments as task8_per_image_mean_v.csv.
Reports AP50 and AP50:95 per tertile for all six runs.
"""

import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Journal-Project")
from ultralytics.utils.metrics import ap_per_class

OUT_DIR = Path(r"D:\Journal-Project\results_v6\journal_eval")
CACHE_DIR = OUT_DIR / "a1_cache"

DATASETS = ["marine_litter", "floating_waste", "flow_img"]
MODELS = [
    "YOLO11n_seed42", "YOLO11n_seed0", "YOLO11n_seed1",
    "FA_EvYOLO_RunB_seed42", "FA_EvYOLO_RunB_seed0", "FA_EvYOLO_RunB_seed1",
]


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


def main():
    print("=" * 80)
    print("TASK A3: Brightness-tertile split with validated per-image stats")
    print("=" * 80)

    # Load the existing tertile assignments
    v_df = pd.read_csv(Path(r"D:\Journal-Project\results_v6\final_audit\task8_per_image_mean_v.csv"))
    print(f"Loaded {len(v_df)} rows from task8_per_image_mean_v.csv")

    rows = []

    for ds in DATASETS:
        ds_v = v_df[v_df["dataset"] == ds]
        for tertile in ["Dark", "Normal", "Bright"]:
            t_imgs = set(ds_v[ds_v["tertile"] == tertile]["image_name"])
            t_sub = ds_v[ds_v["tertile"] == tertile]
            v_min = float(t_sub["mean_V"].min())
            v_max = float(t_sub["mean_V"].max())
            n_imgs = len(t_imgs)

            for model_name in MODELS:
                per_image = load_per_image(model_name, ds)
                # Filter to tertile images
                sub_stats = [per_image[fn] for fn in per_image.keys() if fn in t_imgs]
                ap50, ap50_95 = compute_ap_from_stats(sub_stats)

                if "YOLO11n" in model_name:
                    family = "YOLO11n"
                else:
                    family = "FA-EvYOLO_RunB"
                seed = int(model_name.split("seed")[-1])

                rows.append({
                    "dataset": ds,
                    "illumination_tertile": tertile,
                    "v_min": v_min,
                    "v_max": v_max,
                    "image_count": n_imgs,
                    "model_family": family,
                    "model_seed": seed,
                    "model_name": model_name,
                    "ap50": ap50,
                    "ap50_95": ap50_95,
                })
                print(f"  {ds:15s} | {tertile:7s} | {model_name:25s} -> AP50={ap50:.6f}, AP50:95={ap50_95:.6f}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "a_illumination.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
