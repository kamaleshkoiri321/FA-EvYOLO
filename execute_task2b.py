"""
execute_task2b.py - Task 2b: Latency Re-Run with Interleaved Randomized Timing
Protocol:
- Device: RTX 4060 Laptop GPU (CUDA:0), FP32, batch 1, 640x640, same real image tensor
- Models loaded once and fused:
  - YOLO11n MuSGD seed 42 (results_v6/A1/train_yolo11n_musgd/weights/best.pt)
  - FA-EvYOLO Run B seed 42 (results_v6/A2/train_fa_evyolo_remapped_seed42/weights/best.pt)
- 200 warm-up iterations per stage
- For i in 1..300, time each of the 5 stages once in random order (seed 0)
- torch.cuda.synchronize() immediately before t0 and after t1
- Repeat 3 times (3 blocks of 300 randomized trials -> 900 timings per stage)
- Reports median and IQR per repeat (R1, R2, R3) and median + IQR over all 900 timings
- Saves CSVs to results_v6/final_audit/
"""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.ops as tv_ops

from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from fa_evyolo_modules import SpectralAmplitudeNorm2d, EvidentialRegularizer, ReflectionUncertaintyNMS

# Register module
for _ns in [ul_modules, ul_tasks]:
    setattr(_ns, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d
if hasattr(ul_tasks, "base_modules"):
    ul_tasks.base_modules = tuple(
        m for m in ul_tasks.base_modules if m.__name__ != "SpectralAmplitudeNorm2d"
    ) + (SpectralAmplitudeNorm2d,)

OUT_DIR = Path(r"D:\Journal-Project\results_v6\final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA device required for Task 2b latency benchmarking!"

gpu_name = torch.cuda.get_device_name(0)
print(f"Device: {gpu_name}")

BASE_WEIGHTS = Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")
FA_WEIGHTS   = Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")

print(f"Base Weights: {BASE_WEIGHTS}")
print(f"FA Weights:   {FA_WEIGHTS}")

# Load and fuse models once
base_model = YOLO(str(BASE_WEIGHTS))
fa_model   = YOLO(str(FA_WEIGHTS))

base_model.model.fuse()
fa_model.model.fuse()

base_net = base_model.model.to(device).eval()
fa_net   = fa_model.model.to(device).eval()

# Load real input image (640x640 letterboxed)
IMG_PATH = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\marine_litter\test\images\image_20241006_150643_046395_jpg.rf.8dacd91c2919954429f8b33fa286e463.jpg")
img_bgr = cv2.imread(str(IMG_PATH))
assert img_bgr is not None, f"Failed to load image at {IMG_PATH}"
h0, w0 = img_bgr.shape[:2]
r = min(640 / h0, 640 / w0)
new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
dw, dh = (640 - new_w) / 2.0, (640 - new_h) / 2.0
img_res = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
img_pad = cv2.copyMakeBorder(img_res, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(device, dtype=torch.float32) / 255.0

print(f"Input tensor: {input_tensor.shape}, device={input_tensor.device}")

# Hooks for FA-EvYOLO full pipeline
cv2_buffers, cv3_buffers = [], []
detect_head = fa_net.model[-1]
for scale_idx in range(3):
    detect_head.cv2[scale_idx].register_forward_hook(lambda m, inp, out, b=cv2_buffers: b.append(out) or None)
    detect_head.cv3[scale_idx].register_forward_hook(lambda m, inp, out, b=cv3_buffers: b.append(out) or None)

ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65).to(device)

CONF_THRES = 0.15
IOU_THRES = 0.65

# Define stage runners
def run_stage_a():
    return base_net(input_tensor)

def run_stage_b():
    preds = base_net(input_tensor)
    p = preds[0][0]
    scores = p[4]
    mask = scores >= CONF_THRES
    if mask.any():
        boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                             p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
        _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)

def run_stage_c():
    cv2_buffers.clear(); cv3_buffers.clear()
    return fa_net(input_tensor)

def run_stage_d():
    cv2_buffers.clear(); cv3_buffers.clear()
    preds = fa_net(input_tensor)
    p = preds[0][0]
    scores = p[4]
    mask = scores >= CONF_THRES
    if mask.any():
        boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                             p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
        _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)

def run_stage_e():
    cv2_buffers.clear(); cv3_buffers.clear()
    preds = fa_net(input_tensor)
    pred_tensor = preds[0][0]
    dfl_flat = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
    cls_flat = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
    scores = pred_tensor[4]
    cand_mask = scores >= CONF_THRES
    if cand_mask.any():
        cx, cy = pred_tensor[0, cand_mask], pred_tensor[1, cand_mask]
        bw, bh = pred_tensor[2, cand_mask], pred_tensor[3, cand_mask]
        boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
        cand_scores = scores[cand_mask]
        unc = EvidentialRegularizer.extract_box_uncertainty(cls_flat[cand_mask, 0], dfl_flat[cand_mask], reg_max=16)
        glare_d = ru_nms.compute_box_glare_density(input_tensor[0], boxes_xyxy)
        mod_scores = cand_scores * torch.exp(-0.35 * unc["u_total"] - 0.65 * glare_d * unc["u_total"])
        _ = tv_ops.nms(boxes_xyxy, mod_scores, IOU_THRES)

STAGE_FUNCS = {
    "a": ("(a) YOLO11n forward", run_stage_a),
    "b": ("(b) YOLO11n forward + standard NMS", run_stage_b),
    "c": ("(c) FA-EvYOLO run B forward", run_stage_c),
    "d": ("(d) FA-EvYOLO forward + standard NMS", run_stage_d),
    "e": ("(e) FA-EvYOLO full pipeline", run_stage_e),
}

# 1. Warm-up: 200 iterations per stage
print("\n[Warmup] Running 200 warm-up iterations per stage...")
with torch.no_grad():
    for stage_key in ["a", "b", "c", "d", "e"]:
        name, fn = STAGE_FUNCS[stage_key]
        for _ in range(200):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
print("Warm-up complete.")

# 2. Main benchmarking loop: 3 repeats of 300 randomized trials
print("\n[Benchmarking] Starting 3 repeats x 300 randomized trials (seed 0)...")
rng = np.random.default_rng(0)

# Storage: {stage_key: {repeat_idx: [300 floats]}}
timings = {k: {r: [] for r in [1, 2, 3]} for k in STAGE_FUNCS.keys()}
raw_log_rows = []

with torch.no_grad():
    for r in [1, 2, 3]:
        t_rep0 = time.time()
        for trial in range(1, 301):
            perm = rng.permutation(["a", "b", "c", "d", "e"])
            for stage_key in perm:
                _, fn = STAGE_FUNCS[stage_key]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                timings[stage_key][r].append(elapsed_ms)
                raw_log_rows.append({
                    "repeat": r,
                    "trial": trial,
                    "stage_key": stage_key,
                    "stage_name": STAGE_FUNCS[stage_key][0],
                    "latency_ms": elapsed_ms
                })
        print(f"  Finished Repeat {r} in {time.time() - t_rep0:.2f}s")

# Save raw 900 iterations per stage CSV
df_raw = pd.DataFrame(raw_log_rows)
raw_csv_path = OUT_DIR / "task2b_latency_raw_900_iterations.csv"
df_raw.to_csv(raw_csv_path, index=False)
print(f"\nSaved raw 900 timings per stage to {raw_csv_path}")

# Calculate metrics per stage
def get_iqr(vals):
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    return float(np.median(vals)), q25, q75, float(q75 - q25)

summary_rows = []
for k, (stage_name, _) in STAGE_FUNCS.items():
    r1_vals = np.array(timings[k][1])
    r2_vals = np.array(timings[k][2])
    r3_vals = np.array(timings[k][3])
    all_vals = np.concatenate([r1_vals, r2_vals, r3_vals])

    m1, q25_1, q75_1, iqr1 = get_iqr(r1_vals)
    m2, q25_2, q75_2, iqr2 = get_iqr(r2_vals)
    m3, q25_3, q75_3, iqr3 = get_iqr(r3_vals)
    m_all, q25_all, q75_all, iqr_all = get_iqr(all_vals)

    fps_overall = 1000.0 / m_all if m_all > 0 else 0.0

    summary_rows.append({
        "stage": stage_name,
        "repeat1_median_ms": m1,
        "repeat1_iqr_ms": iqr1,
        "repeat1_q25_ms": q25_1,
        "repeat1_q75_ms": q75_1,
        "repeat2_median_ms": m2,
        "repeat2_iqr_ms": iqr2,
        "repeat2_q25_ms": q25_2,
        "repeat2_q75_ms": q75_2,
        "repeat3_median_ms": m3,
        "repeat3_iqr_ms": iqr3,
        "repeat3_q25_ms": q25_3,
        "repeat3_q75_ms": q75_3,
        "overall_900_median_ms": m_all,
        "overall_900_iqr_ms": iqr_all,
        "overall_900_q25_ms": q25_all,
        "overall_900_q75_ms": q75_all,
        "overall_900_mean_ms": float(np.mean(all_vals)),
        "overall_900_std_ms": float(np.std(all_vals)),
        "overall_900_min_ms": float(np.min(all_vals)),
        "overall_900_max_ms": float(np.max(all_vals)),
        "overall_fps_at_median": fps_overall,
        "gpu_hardware": gpu_name,
        "batch_size": 1,
        "precision": "FP32",
        "random_seed": 0
    })

df_summary = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "task2b_latency_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"Saved summary latency table to {summary_csv}")

print("\n" + "="*95)
print("TASK 2b LATENCY RE-RUN RESULTS (Unrounded)")
print("="*95)
for _, r in df_summary.iterrows():
    print(f"\n{r['stage']}:")
    print(f"  Repeat 1 (N=300): Median = {r['repeat1_median_ms']:.6f} ms | IQR = {r['repeat1_iqr_ms']:.6f} ms (Q1={r['repeat1_q25_ms']:.6f}, Q3={r['repeat1_q75_ms']:.6f})")
    print(f"  Repeat 2 (N=300): Median = {r['repeat2_median_ms']:.6f} ms | IQR = {r['repeat2_iqr_ms']:.6f} ms (Q1={r['repeat2_q25_ms']:.6f}, Q3={r['repeat2_q75_ms']:.6f})")
    print(f"  Repeat 3 (N=300): Median = {r['repeat3_median_ms']:.6f} ms | IQR = {r['repeat3_iqr_ms']:.6f} ms (Q1={r['repeat3_q25_ms']:.6f}, Q3={r['repeat3_q75_ms']:.6f})")
    print(f"  --> OVERALL (N=900): Median = {r['overall_900_median_ms']:.6f} ms | IQR = {r['overall_900_iqr_ms']:.6f} ms | FPS = {r['overall_fps_at_median']:.2f}")
    print(f"      Mean +- Std = {r['overall_900_mean_ms']:.6f} +- {r['overall_900_std_ms']:.6f} ms | Min = {r['overall_900_min_ms']:.6f} ms, Max = {r['overall_900_max_ms']:.6f} ms")

