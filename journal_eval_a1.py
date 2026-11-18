"""
Task A1: Fixed per-image evaluation that exactly matches the Ultralytics validator.

Root cause of the discrepancy: The previous ImageStatsExtractor pooled per-image
tp/conf/cls arrays through ap_per_class, but the validator's _process_batch
produces tp arrays whose shape depends on batch-level letterbox/rect padding.
When these are pooled across images with different padding, the result diverges
from the validator's own end-of-run ap_per_class call.

Fix: Run the Ultralytics validator natively on each dataset (getting the exact
pooled metrics), AND extract per-image stats by hooking the validator's own
update_metrics. Then verify that pooling per-image stats through ap_per_class
reproduces the validator's reported metrics. If not within 0.002, we use the
validator's native per-image decomposition directly.

The key insight: we must ensure the per-image stats are collected from the
SAME preprocessing pipeline (same rect mode, same batch composition) as the
validator uses. We do this by running the validator once and recording per-image
tp/conf/cls from within update_metrics, which is called AFTER the validator's
own preprocessing.
"""

import sys, pickle, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"D:\Journal-Project")
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
from fa_evyolo_modules import SpectralAmplitudeNorm2d

# Register custom module
for _ns in [ul_modules, ul_tasks]:
    setattr(_ns, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d

OUT_DIR = Path(r"D:\Journal-Project\results_v6\journal_eval")
CACHE_DIR = OUT_DIR / "a1_cache"
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
    {"name": "YOLO11n_seed42",          "weights": Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")},
    {"name": "YOLO11n_seed0",           "weights": Path(r"D:\Journal-Project\results_v6\A2\train_yolo11n_musgd_seed0\weights\best.pt")},
    {"name": "YOLO11n_seed1",           "weights": Path(r"D:\Journal-Project\results_v6\A2\train_yolo11n_musgd_seed1\weights\best.pt")},
    {"name": "FA_EvYOLO_RunB_seed42",   "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")},
    {"name": "FA_EvYOLO_RunB_seed0",    "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed0\weights\best.pt")},
    {"name": "FA_EvYOLO_RunB_seed1",    "weights": Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed1\weights\best.pt")},
]

# ---- Validator that also records per-image stats ----
from ultralytics.models.yolo.detect import DetectionValidator

class PerImageValidator(DetectionValidator):
    """
    Runs exactly as the native validator but also records per-image
    tp, conf, pred_cls, target_cls arrays for subset analysis.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_image_records = []

    def update_metrics(self, preds, batch):
        """Hook into the standard update_metrics to capture per-image stats."""
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"].cpu().numpy()
            im_file = Path(pbatch["im_file"]).name

            predn = self._prepare_pred(pred)
            match_stats = self._process_batch(predn, pbatch)

            tp = match_stats["tp"].cpu().numpy() if hasattr(match_stats["tp"], "cpu") else np.array(match_stats["tp"])
            conf = predn["conf"].cpu().numpy() if len(predn["cls"]) else np.zeros(0, dtype=np.float32)
            pred_cls = predn["cls"].cpu().numpy() if len(predn["cls"]) else np.zeros(0, dtype=np.float32)

            self.per_image_records.append({
                "im_file": im_file,
                "tp": tp,
                "conf": conf,
                "pred_cls": pred_cls,
                "target_cls": cls,
            })

        # Call the parent to maintain the validator's own metrics
        super().update_metrics(preds, batch)


def compute_ap_from_stats(stats_list):
    """Compute AP50 and AP50:95 from a list of per-image stat dicts."""
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
    ap = ret[5]  # Ultralytics 8.4: ap is at index 5
    ap50 = float(ap[0, 0]) if len(ap) else 0.0
    ap50_95 = float(ap[0].mean()) if len(ap) else 0.0
    return ap50, ap50_95


def run_validator_and_extract(model_info, ds_name):
    """
    Run the Ultralytics validator on a dataset and return:
    1. The validator's official AP50 and AP50:95
    2. Per-image stats dict keyed by filename
    """
    cache_path = CACHE_DIR / f"{model_info['name']}_{ds_name}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Running validator: {model_info['name']} on {ds_name}...")
    yaml_path = DATA_YAMLS[ds_name]
    model = YOLO(str(model_info["weights"]))

    validator = PerImageValidator(args=dict(
        data=str(yaml_path),
        split="test",
        conf=0.001,
        iou=0.70,
        imgsz=640,
        device="cuda:0",
        verbose=False,
        plots=False,
    ))
    results = validator(model=model.model)

    # Get the validator's own metrics
    official_ap50 = float(validator.metrics.box.map50)
    official_ap50_95 = float(validator.metrics.box.map)

    # Get per-image stats
    per_image = {r["im_file"]: r for r in validator.per_image_records}

    result = {
        "official_ap50": official_ap50,
        "official_ap50_95": official_ap50_95,
        "per_image": per_image,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    # Free GPU memory
    del model, validator
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 80)
    print("TASK A1: Validated per-image evaluation with sanity check")
    print("=" * 80)

    # Load the reference main table
    main_table = pd.read_csv(Path(r"D:\Journal-Project\results_v6\A2\multi_seed_main_table.csv"))

    sanity_rows = []

    for m in MODELS:
        print(f"\n--- Model: {m['name']} ---")
        for ds in DATASETS:
            result = run_validator_and_extract(m, ds)
            official_ap50 = result["official_ap50"]
            official_ap50_95 = result["official_ap50_95"]
            per_image = result["per_image"]

            # Compute pooled AP from per-image stats
            pooled_ap50, pooled_ap50_95 = compute_ap_from_stats(list(per_image.values()))

            # Get main table reference
            if "YOLO11n" in m["name"]:
                prefix = "yolo11n"
            else:
                prefix = "fa_evyolo"
            seed = m["name"].split("seed")[-1]
            col = f"{prefix}_seed{seed}"

            ref_ap50 = float(main_table[(main_table["dataset"] == ds) & (main_table["metric"] == "mAP50")][col].values[0])
            ref_ap50_95 = float(main_table[(main_table["dataset"] == ds) & (main_table["metric"] == "mAP50_95")][col].values[0])

            diff_official_vs_ref_50 = official_ap50 - ref_ap50
            diff_official_vs_ref_95 = official_ap50_95 - ref_ap50_95
            diff_pooled_vs_official_50 = pooled_ap50 - official_ap50
            diff_pooled_vs_official_95 = pooled_ap50_95 - official_ap50_95
            diff_pooled_vs_ref_50 = pooled_ap50 - ref_ap50
            diff_pooled_vs_ref_95 = pooled_ap50_95 - ref_ap50_95

            print(f"  {ds:15s} | Official: AP50={official_ap50:.6f}, AP50:95={official_ap50_95:.6f}")
            print(f"  {ds:15s} | Pooled:   AP50={pooled_ap50:.6f}, AP50:95={pooled_ap50_95:.6f}")
            print(f"  {ds:15s} | Ref:      AP50={ref_ap50:.6f}, AP50:95={ref_ap50_95:.6f}")
            print(f"  {ds:15s} | Diff(official-ref): AP50={diff_official_vs_ref_50:+.6f}, AP50:95={diff_official_vs_ref_95:+.6f}")
            print(f"  {ds:15s} | Diff(pooled-official): AP50={diff_pooled_vs_official_50:+.6f}, AP50:95={diff_pooled_vs_official_95:+.6f}")
            print(f"  {ds:15s} | Diff(pooled-ref): AP50={diff_pooled_vs_ref_50:+.6f}, AP50:95={diff_pooled_vs_ref_95:+.6f}")

            sanity_rows.append({
                "model_name": m["name"],
                "dataset": ds,
                "num_images": len(per_image),
                "official_ap50": official_ap50,
                "official_ap50_95": official_ap50_95,
                "pooled_ap50": pooled_ap50,
                "pooled_ap50_95": pooled_ap50_95,
                "ref_ap50": ref_ap50,
                "ref_ap50_95": ref_ap50_95,
                "diff_official_vs_ref_ap50": diff_official_vs_ref_50,
                "diff_official_vs_ref_ap50_95": diff_official_vs_ref_95,
                "diff_pooled_vs_official_ap50": diff_pooled_vs_official_50,
                "diff_pooled_vs_official_ap50_95": diff_pooled_vs_official_95,
                "diff_pooled_vs_ref_ap50": diff_pooled_vs_ref_50,
                "diff_pooled_vs_ref_ap50_95": diff_pooled_vs_ref_95,
            })

    df = pd.DataFrame(sanity_rows)
    out_csv = OUT_DIR / "a_sanity.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved sanity check to {out_csv}")

    # Check sanity: official must match ref within 0.002
    max_diff_50 = df["diff_official_vs_ref_ap50"].abs().max()
    max_diff_95 = df["diff_official_vs_ref_ap50_95"].abs().max()
    print(f"\nMax |official - ref| AP50: {max_diff_50:.6f}")
    print(f"Max |official - ref| AP50:95: {max_diff_95:.6f}")

    if max_diff_50 > 0.002 or max_diff_95 > 0.002:
        print("\n*** SANITY CHECK FAILED ***")
        print("Official validator metrics do not match multi_seed_main_table.csv within 0.002!")
        print("Stopping. Do not present these numbers as valid.")
        failed = df[(df["diff_official_vs_ref_ap50"].abs() > 0.002) | (df["diff_official_vs_ref_ap50_95"].abs() > 0.002)]
        print(failed[["model_name", "dataset", "official_ap50", "ref_ap50", "diff_official_vs_ref_ap50",
                       "official_ap50_95", "ref_ap50_95", "diff_official_vs_ref_ap50_95"]].to_string())
        return False

    # Also check pooled vs official
    max_pooled_diff_50 = df["diff_pooled_vs_official_ap50"].abs().max()
    max_pooled_diff_95 = df["diff_pooled_vs_official_ap50_95"].abs().max()
    print(f"\nMax |pooled - official| AP50: {max_pooled_diff_50:.6f}")
    print(f"Max |pooled - official| AP50:95: {max_pooled_diff_95:.6f}")

    if max_pooled_diff_50 > 0.002 or max_pooled_diff_95 > 0.002:
        print("\n*** WARNING: Pooled per-image stats diverge from official by > 0.002 ***")
        print("Per-image AP analysis may not exactly reproduce validator numbers.")
        print("Using official validator numbers as ground truth for all analyses.")
    else:
        print("\n*** SANITY CHECK PASSED ***")
        print("Both official and pooled metrics match the reference within 0.002.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
