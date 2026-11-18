"""
execute_task4.py - Task 4: AP Confidence Intervals via Paired Bootstrap
- Evaluates Seed 42 YOLO11n vs Run B Seed 42 under Ultralytics validator settings:
  conf=0.001, iou=0.70, imgsz=640.
- Paired bootstrap over test images (10,000 resamples, random seed 0).
- On FloW-Img, resample by source image (filename prefix before '_jpg').
- Reports observed AP50 and AP50:95 differences with 95% confidence intervals
  [2.5%, 97.5%] for Floating Waste and FloW-Img.
- Outputs CSVs to results_v6/final_audit/.
"""

import sys, time, pickle
from pathlib import Path
from collections import defaultdict
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
CACHE_DIR = OUT_DIR / "task4_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_WEIGHTS = Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")
FA_WEIGHTS   = Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")

DATA_YAMLS = {
    "floating_waste": Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\floating_waste\floating_waste.yaml"),
    "flow_img":       Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\flow_img\flow_img.yaml"),
}

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


def run_extraction(weights_path, data_yaml):
    cache_file = CACHE_DIR / f"{data_yaml.stem}_{weights_path.parent.parent.name}_{weights_path.stem}.pkl"
    if cache_file.exists():
        print(f"Loading cached validator stats from: {cache_file.name}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Extracting validator stats for: {weights_path.name} on {data_yaml.stem}...")
    model = YOLO(str(weights_path))
    validator = ImageStatsExtractor(args=dict(
        data=str(data_yaml),
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
    with open(cache_file, "wb") as f:
        pickle.dump(res, f)
    return res


def compute_dataset_ap(stats_list):
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
    N_BOOTSTRAP = 10000
    SEED = 0
    rng = np.random.default_rng(SEED)

    ci_summary_records = []
    raw_diffs_records = {}

    for ds_name, yaml_path in DATA_YAMLS.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING DATASET: {ds_name}")
        print(f"{'='*70}")

        base_dict = run_extraction(BASE_WEIGHTS, yaml_path)
        fa_dict   = run_extraction(FA_WEIGHTS, yaml_path)

        # Align images identically
        common_imgs = sorted(list(set(base_dict.keys()) & set(fa_dict.keys())))
        print(f"Aligned {len(common_imgs)} common test images.")

        base_aligned = [base_dict[fn] for fn in common_imgs]
        fa_aligned   = [fa_dict[fn] for fn in common_imgs]

        # Observed AP on full test set
        base_ap50_obs, base_ap50_95_obs = compute_dataset_ap(base_aligned)
        fa_ap50_obs,   fa_ap50_95_obs   = compute_dataset_ap(fa_aligned)

        diff_ap50_obs    = fa_ap50_obs - base_ap50_obs
        diff_ap50_95_obs = fa_ap50_95_obs - base_ap50_95_obs

        print(f"Observed Base:  AP50 = {base_ap50_obs:.6f}, AP50:95 = {base_ap50_95_obs:.6f}")
        print(f"Observed FA:    AP50 = {fa_ap50_obs:.6f}, AP50:95 = {fa_ap50_95_obs:.6f}")
        print(f"Observed Diff:  dAP50 = {diff_ap50_obs:+.6f}, dAP50:95 = {diff_ap50_95_obs:+.6f}")

        # Setup clusters for resampling
        if ds_name == "flow_img":
            resample_unit_desc = "Cluster (source image prefix before '_jpg')"
            clusters = defaultdict(list)
            for idx, fn in enumerate(common_imgs):
                # e.g. 000293_jpg.rf.eac... -> source prefix is 000293
                src_prefix = fn.split("_jpg")[0]
                clusters[src_prefix].append(idx)
            cluster_keys = sorted(list(clusters.keys()))
            n_clusters = len(cluster_keys)
            print(f"Grouped {len(common_imgs)} FloW-Img images into {n_clusters} source image clusters.")
        else:
            resample_unit_desc = "Individual test image"
            cluster_keys = list(range(len(common_imgs)))
            clusters = {i: [i] for i in range(len(common_imgs))}
            n_clusters = len(cluster_keys)
            print(f"Individual image resampling across {n_clusters} images.")

        boot_d_ap50 = []
        boot_d_ap50_95 = []

        print(f"\nRunning {N_BOOTSTRAP:,} paired bootstrap iterations (seed {SEED})...")
        t0 = time.time()
        for b in range(N_BOOTSTRAP):
            # Resample clusters with replacement
            sampled_keys = rng.choice(cluster_keys, size=n_clusters, replace=True)
            sampled_indices = []
            for k in sampled_keys:
                sampled_indices.extend(clusters[k])

            b_base_sample = [base_aligned[i] for i in sampled_indices]
            b_fa_sample   = [fa_aligned[i] for i in sampled_indices]

            b_b50, b_b95 = compute_dataset_ap(b_base_sample)
            b_f50, b_f95 = compute_dataset_ap(b_fa_sample)

            boot_d_ap50.append(b_f50 - b_b50)
            boot_d_ap50_95.append(b_f95 - b_b95)

            if (b + 1) % 2000 == 0:
                print(f"  Iteration {b+1:,}/{N_BOOTSTRAP:,} ({time.time() - t0:.1f}s)...")

        elapsed = time.time() - t0
        print(f"Completed {N_BOOTSTRAP:,} iterations in {elapsed:.1f}s ({elapsed/N_BOOTSTRAP*1000:.2f} ms/iter).")

        boot_d_ap50 = np.array(boot_d_ap50)
        boot_d_ap50_95 = np.array(boot_d_ap50_95)

        ci_ap50_low, ci_ap50_high = float(np.percentile(boot_d_ap50, 2.5)), float(np.percentile(boot_d_ap50, 97.5))
        ci_ap95_low, ci_ap95_high = float(np.percentile(boot_d_ap50_95, 2.5)), float(np.percentile(boot_d_ap50_95, 97.5))

        raw_diffs_records[f"{ds_name}_diff_ap50"] = boot_d_ap50
        raw_diffs_records[f"{ds_name}_diff_ap50_95"] = boot_d_ap50_95

        ci_summary_records.append({
            "dataset": ds_name,
            "resampling_unit": resample_unit_desc,
            "n_units": n_clusters,
            "total_images": len(common_imgs),
            "observed_base_ap50": base_ap50_obs,
            "observed_fa_ap50": fa_ap50_obs,
            "observed_diff_ap50": diff_ap50_obs,
            "ci95_lower_diff_ap50": ci_ap50_low,
            "ci95_upper_diff_ap50": ci_ap50_high,
            "observed_base_ap50_95": base_ap50_95_obs,
            "observed_fa_ap50_95": fa_ap50_95_obs,
            "observed_diff_ap50_95": diff_ap50_95_obs,
            "ci95_lower_diff_ap50_95": ci_ap95_low,
            "ci95_upper_diff_ap50_95": ci_ap95_high,
            "bootstrap_samples": N_BOOTSTRAP,
            "seed": SEED,
            "conf_eval": 0.001,
            "iou_eval": 0.70,
            "base_weights": str(BASE_WEIGHTS),
            "fa_weights": str(FA_WEIGHTS)
        })

    # Save summary CSV
    df_summary = pd.DataFrame(ci_summary_records)
    summary_path = OUT_DIR / "task4_bootstrap_ap_ci.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary bootstrap CIs to {summary_path}")

    # Save raw differences
    df_raw = pd.DataFrame(raw_diffs_records)
    raw_path = OUT_DIR / "task4_bootstrap_differences_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw bootstrap distribution to {raw_path}")

    print("\n" + "="*80)
    print("TASK 4 AP CONFIDENCE INTERVAL RESULTS (Unrounded)")
    print("="*80)
    for _, r in df_summary.iterrows():
        print(f"\nDataset: {r['dataset']} ({r['resampling_unit']}, N={r['n_units']})")
        print(f"  Observed AP50:    YOLO11n = {r['observed_base_ap50']:.6f} | FA-EvYOLO = {r['observed_fa_ap50']:.6f}")
        print(f"  Difference AP50:   {r['observed_diff_ap50']:+.6f} (95% CI: [{r['ci95_lower_diff_ap50']:+.6f}, {r['ci95_upper_diff_ap50']:+.6f}])")
        print(f"  Observed AP50:95: YOLO11n = {r['observed_base_ap50_95']:.6f} | FA-EvYOLO = {r['observed_fa_ap50_95']:.6f}")
        print(f"  Difference AP50:95:{r['observed_diff_ap50_95']:+.6f} (95% CI: [{r['ci95_lower_diff_ap50_95']:+.6f}, {r['ci95_upper_diff_ap50_95']:+.6f}])")

if __name__ == "__main__":
    main()
