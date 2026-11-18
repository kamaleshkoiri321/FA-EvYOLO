"""
execute_task8.py - Task 8: Illumination Tertile Split (Dark / Normal / Bright)
1. For every test image across the three datasets (marine_litter, floating_waste, flow_img),
   computes the mean HSV Value (V, 0-255).
2. Within each dataset, splits test images into brightness tertiles (Dark, Normal, Bright)
   using 33.333% and 66.667% quantiles.
3. Reports per tertile:
   - V range (min, max)
   - Image count
   - Validator AP50 and AP50:95 (conf 0.001, IoU 0.7) for all 6 multi-seed runs:
     * YOLO11n seed 42, seed 0, seed 1
     * FA-EvYOLO Run B seed 42, seed 0, seed 1
4. Saves per-image mean V to results_v6/final_audit/task8_per_image_mean_v.csv
   and summary tertile evaluation table to results_v6/final_audit/task8_illumination_tertiles_summary.csv.
"""

import sys, time, pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"D:\Journal-Project")
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class
from fa_evyolo_modules import SpectralAmplitudeNorm2d

# Register custom module
for _ns in [ul_modules, ul_tasks]:
    setattr(_ns, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d
if hasattr(ul_tasks, "base_modules"):
    ul_tasks.base_modules = tuple(
        m for m in ul_tasks.base_modules if m.__name__ != "SpectralAmplitudeNorm2d"
    ) + (SpectralAmplitudeNorm2d,)

OUT_DIR = Path(r"D:\Journal-Project\results_v6\final_audit")
CACHE_DIR = OUT_DIR / "task8_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark")
DATASETS = ["marine_litter", "floating_waste", "flow_img"]

DATA_YAMLS = {
    "marine_litter":  BENCHMARK / "marine_litter"  / "marine_litter.yaml",
    "floating_waste": BENCHMARK / "floating_waste" / "floating_waste.yaml",
    "flow_img":       BENCHMARK / "flow_img"       / "flow_img.yaml",
}

MODELS = [
    {
        "model_family": "YOLO11n",
        "seed": 42,
        "name": "YOLO11n_seed42",
        "weights": Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")
    },
    {
        "model_family": "YOLO11n",
        "seed": 0,
        "name": "YOLO11n_seed0",
        "weights": Path(r"D:\Journal-Project\results_v6\A2\train_yolo11n_musgd_seed0\weights\best.pt")
    },
    {
        "model_family": "YOLO11n",
        "seed": 1,
        "name": "YOLO11n_seed1",
        "weights": Path(r"D:\Journal-Project\results_v6\A2\train_yolo11n_musgd_seed1\weights\best.pt")
    },
    {
        "model_family": "FA-EvYOLO_RunB",
        "seed": 42,
        "name": "FA_EvYOLO_RunB_seed42",
        "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")
    },
    {
        "model_family": "FA-EvYOLO_RunB",
        "seed": 0,
        "name": "FA_EvYOLO_RunB_seed0",
        "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed0\weights\best.pt")
    },
    {
        "model_family": "FA-EvYOLO_RunB",
        "seed": 1,
        "name": "FA_EvYOLO_RunB_seed1",
        "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed1\weights\best.pt")
    },
]

# ---------------------------------------------------------------------------
# 1. Compute Mean HSV V for all test images & split into tertiles
# ---------------------------------------------------------------------------
def compute_illumination_tertiles():
    records = []
    for ds in DATASETS:
        img_dir = BENCHMARK / ds / "test" / "images"
        files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Reading {len(files)} test images for {ds}...")
        v_list = []
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_v = float(np.mean(hsv[:, :, 2]))
            v_list.append((p.name, mean_v))

        # Sort and tertile split
        df_ds = pd.DataFrame(v_list, columns=["image_name", "mean_V"])
        q33 = float(np.percentile(df_ds["mean_V"], 100.0 / 3.0))
        q66 = float(np.percentile(df_ds["mean_V"], 200.0 / 3.0))

        tertiles = []
        for _, r in df_ds.iterrows():
            v = r["mean_V"]
            if v <= q33:
                t = "Dark"
            elif v <= q66:
                t = "Normal"
            else:
                t = "Bright"
            tertiles.append(t)
        df_ds["tertile"] = tertiles
        df_ds["dataset"] = ds
        df_ds["q33"] = q33
        df_ds["q66"] = q66
        records.append(df_ds)

    df_all = pd.concat(records, ignore_index=True)
    out_csv = OUT_DIR / "task8_per_image_mean_v.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved per-image mean V to {out_csv} ({len(df_all)} images total).")
    return df_all

# ---------------------------------------------------------------------------
# 2. Extract validator predictions for all 6 models across 3 datasets
# ---------------------------------------------------------------------------
class ImageStatsExtractor(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted_records = []

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"].cpu().numpy()
            predn = self._prepare_pred(pred)
            match_stats = self._process_batch(predn, pbatch)
            tp = match_stats["tp"].cpu().numpy() if hasattr(match_stats["tp"], "cpu") else np.array(match_stats["tp"])
            conf = predn["conf"].cpu().numpy() if len(predn["cls"]) else np.zeros(0, dtype=np.float32)
            pred_cls = predn["cls"].cpu().numpy() if len(predn["cls"]) else np.zeros(0, dtype=np.float32)

            self.extracted_records.append({
                "im_file": Path(pbatch["im_file"]).name,
                "tp": tp,
                "conf": conf,
                "pred_cls": pred_cls,
                "target_cls": cls,
            })
        super().update_metrics(preds, batch)

def extract_or_load_stats(model_info, ds_name):
    cache_path = CACHE_DIR / f"{model_info['name']}_{ds_name}_stats.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Extracting {model_info['name']} on {ds_name}...")
    yaml_path = DATA_YAMLS[ds_name]
    model = YOLO(str(model_info["weights"]))
    validator = ImageStatsExtractor(args=dict(
        data=str(yaml_path),
        split="test",
        conf=0.001,
        iou=0.70,
        imgsz=640,
        device="cuda:0",
        verbose=False,
        plots=False,
    ))
    validator(model=model.model)
    res = {r["im_file"]: r for r in validator.extracted_records}
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    return res

def compute_ap_from_stats(stats_list):
    if not stats_list:
        return 0.0, 0.0
    tp_all = np.concatenate([s["tp"] for s in stats_list], axis=0) if sum(len(s["tp"]) for s in stats_list) > 0 else np.zeros((0, 10), dtype=bool)
    conf_all = np.concatenate([s["conf"] for s in stats_list], axis=0) if sum(len(s["conf"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate([s["pred_cls"] for s in stats_list], axis=0) if sum(len(s["pred_cls"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)
    target_cls_all = np.concatenate([s["target_cls"] for s in stats_list], axis=0) if sum(len(s["target_cls"]) for s in stats_list) > 0 else np.zeros((0,), dtype=np.float32)

    if len(target_cls_all) == 0:
        return 0.0, 0.0

    ret = ap_per_class(
        tp_all, conf_all, pred_cls_all, target_cls_all,
        plot=False, save_dir=Path("."), names={0: "item"}
    )
    ap = ret[5]  # in Ultralytics 8.4, ap is index 5
    ap50 = float(ap[0, 0]) if len(ap) else 0.0
    ap50_95 = float(ap[0].mean()) if len(ap) else 0.0
    return ap50, ap50_95

def main():
    print("Step 1: Computing HSV V tertiles...")
    df_v = compute_illumination_tertiles()

    # Summarize tertile boundaries
    tertile_info = {}
    for ds in DATASETS:
        sub = df_v[df_v["dataset"] == ds]
        tertile_info[ds] = {}
        for t in ["Dark", "Normal", "Bright"]:
            t_sub = sub[sub["tertile"] == t]
            tertile_info[ds][t] = {
                "count": len(t_sub),
                "min_v": float(t_sub["mean_V"].min()),
                "max_v": float(t_sub["mean_V"].max()),
                "images": set(t_sub["image_name"])
            }
            print(f"{ds:15s} | {t:7s}: N={len(t_sub):3d} | V range: [{tertile_info[ds][t]['min_v']:.4f}, {tertile_info[ds][t]['max_v']:.4f}]")

    print("\nStep 2: Evaluating 6 models on each tertile subset...")
    summary_records = []

    for m in MODELS:
        print(f"\n--- Model: {m['name']} ---")
        for ds in DATASETS:
            stats_dict = extract_or_load_stats(m, ds)
            for t in ["Dark", "Normal", "Bright"]:
                t_imgs = tertile_info[ds][t]["images"]
                sub_stats = [stats_dict[fn] for fn in stats_dict.keys() if fn in t_imgs]
                ap50, ap50_95 = compute_ap_from_stats(sub_stats)
                summary_records.append({
                    "dataset": ds,
                    "illumination_tertile": t,
                    "v_min": tertile_info[ds][t]["min_v"],
                    "v_max": tertile_info[ds][t]["max_v"],
                    "image_count": tertile_info[ds][t]["count"],
                    "model_family": m["model_family"],
                    "model_seed": m["seed"],
                    "model_name": m["name"],
                    "ap50": ap50,
                    "ap50_95": ap50_95,
                    "weights_path": str(m["weights"])
                })
                print(f"  {ds:15s} | {t:7s} | {m['name']:25s} -> AP50={ap50:.6f}, AP50:95={ap50_95:.6f}")

    df_res = pd.DataFrame(summary_records)
    out_summary_csv = OUT_DIR / "task8_illumination_tertiles_summary.csv"
    df_res.to_csv(out_summary_csv, index=False)
    print(f"\nSaved summary tertile evaluations to {out_summary_csv}")

    # Generate Pivot View for Clean Inspection
    pivot_ap50 = df_res.pivot_table(index=["dataset", "illumination_tertile", "image_count", "v_min", "v_max"],
                                    columns=["model_name"], values=["ap50", "ap50_95"])
    print("\n" + "="*100)
    print("TASK 8 ILLUMINATION SPLIT: AP50 and AP50:95 by Brightness Tertile (Unrounded)")
    print("="*100)
    print(pivot_ap50.to_string())

if __name__ == "__main__":
    main()

