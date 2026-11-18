"""
execute_task2.py - Task 2: Latency Benchmarking (One Session)
Hardware: RTX 4060 Laptop GPU
Dtype: FP32
Batch Size: 1, 640x640
Input: Same real test image (image_20241006_150643_046395_jpg...)
Iterations: 50 warmup + 300 timed iterations
torch.cuda.synchronize() around each timing
Metrics: Median and IQR (Q3 - Q1) in ms without rounding
"""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.ops as tv_ops

from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from fa_evyolo_modules import SpectralAmplitudeNorm2d, EvidentialRegularizer, ReflectionUncertaintyNMS

# Register module
setattr(ul_modules, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
setattr(ul_tasks, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d
if hasattr(ul_tasks, "base_modules"):
    ul_tasks.base_modules = tuple(
        m for m in ul_tasks.base_modules if m.__name__ != "SpectralAmplitudeNorm2d"
    ) + (SpectralAmplitudeNorm2d,)

OUT_DIR = Path(r"D:\Journal-Project\results_v6\final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA device required for Task 2 latency benchmarking!"

gpu_name = torch.cuda.get_device_name(0)
print(f"Device: {gpu_name}")

BASE_WEIGHTS = Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")
FA_WEIGHTS   = Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")

print(f"Base Weights: {BASE_WEIGHTS}")
print(f"FA Weights:   {FA_WEIGHTS}")

# Load and fuse models
base_model = YOLO(str(BASE_WEIGHTS))
fa_model   = YOLO(str(FA_WEIGHTS))

base_model.model.fuse()
fa_model.model.fuse()

base_net = base_model.model.to(device).eval()
fa_net   = fa_model.model.to(device).eval()

# Load real input image
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

print(f"Input image tensor: {input_tensor.shape}, dtype={input_tensor.dtype}, device={input_tensor.device}")

# Hooks for FA-EvYOLO full pipeline
cv2_buffers, cv3_buffers = [], []
detect_head = fa_net.model[-1]
for scale_idx in range(3):
    detect_head.cv2[scale_idx].register_forward_hook(lambda m, inp, out, b=cv2_buffers: b.append(out) or None)
    detect_head.cv3[scale_idx].register_forward_hook(lambda m, inp, out, b=cv3_buffers: b.append(out) or None)

ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65).to(device)

WARMUP = 50
NITERS = 300
CONF_THRES = 0.15
IOU_THRES = 0.65

raw_timings = {}

# ---------------------------------------------------------------------------
# (a) YOLO11n forward
# ---------------------------------------------------------------------------
print("\n[Benchmarking] (a) YOLO11n forward...")
with torch.no_grad():
    for _ in range(WARMUP):
        torch.cuda.synchronize()
        _ = base_net(input_tensor)
        torch.cuda.synchronize()
    times_a = []
    for _ in range(NITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = base_net(input_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_a.append((t1 - t0) * 1000.0)
raw_timings["(a) YOLO11n forward"] = times_a

# ---------------------------------------------------------------------------
# (b) YOLO11n forward + standard NMS
# ---------------------------------------------------------------------------
print("[Benchmarking] (b) YOLO11n forward + standard NMS...")
with torch.no_grad():
    for _ in range(WARMUP):
        torch.cuda.synchronize()
        preds = base_net(input_tensor)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= CONF_THRES
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)
        torch.cuda.synchronize()
    times_b = []
    for _ in range(NITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = base_net(input_tensor)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= CONF_THRES
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_b.append((t1 - t0) * 1000.0)
raw_timings["(b) YOLO11n forward + standard NMS"] = times_b

# ---------------------------------------------------------------------------
# (c) FA-EvYOLO run B forward
# ---------------------------------------------------------------------------
print("[Benchmarking] (c) FA-EvYOLO run B forward...")
with torch.no_grad():
    for _ in range(WARMUP):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
        _ = fa_net(input_tensor)
        torch.cuda.synchronize()
    times_c = []
    for _ in range(NITERS):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fa_net(input_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_c.append((t1 - t0) * 1000.0)
raw_timings["(c) FA-EvYOLO run B forward"] = times_c

# ---------------------------------------------------------------------------
# (d) FA-EvYOLO forward + standard NMS
# ---------------------------------------------------------------------------
print("[Benchmarking] (d) FA-EvYOLO forward + standard NMS...")
with torch.no_grad():
    for _ in range(WARMUP):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
        preds = fa_net(input_tensor)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= CONF_THRES
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)
        torch.cuda.synchronize()
    times_d = []
    for _ in range(NITERS):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = fa_net(input_tensor)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= CONF_THRES
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = tv_ops.nms(boxes, scores[mask], IOU_THRES)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_d.append((t1 - t0) * 1000.0)
raw_timings["(d) FA-EvYOLO forward + standard NMS"] = times_d

# ---------------------------------------------------------------------------
# (e) FA-EvYOLO full pipeline (forward + uncertainty + glare map + RU-NMS)
# ---------------------------------------------------------------------------
print("[Benchmarking] (e) FA-EvYOLO full pipeline...")
with torch.no_grad():
    for _ in range(WARMUP):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
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
        torch.cuda.synchronize()
    times_e = []
    for _ in range(NITERS):
        cv2_buffers.clear(); cv3_buffers.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
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
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_e.append((t1 - t0) * 1000.0)
raw_timings["(e) FA-EvYOLO full pipeline (forward + unc + glare + RU-NMS)"] = times_e

# Save raw iterations CSV
df_raw = pd.DataFrame(raw_timings)
raw_csv_path = OUT_DIR / "task2_latency_raw_iterations.csv"
df_raw.to_csv(raw_csv_path, index=False)
print(f"\nSaved raw iteration times to {raw_csv_path}")

# Summary statistics table
summary_records = []
for col, arr in raw_timings.items():
    arr = np.array(arr)
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = float(q75 - q25)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    fps = 1000.0 / med if med > 0 else 0.0
    summary_records.append({
        "stage": col,
        "median_ms": med,
        "iqr_ms": iqr,
        "q25_ms": q25,
        "q75_ms": q75,
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min_v,
        "max_ms": max_v,
        "fps_at_median": fps,
        "hardware": gpu_name,
        "batch_size": 1,
        "precision": "FP32",
        "warmup_iters": WARMUP,
        "timed_iters": NITERS,
        "seed": SEED,
        "weights_file": str(FA_WEIGHTS) if "FA-EvYOLO" in col else str(BASE_WEIGHTS)
    })

df_summary = pd.DataFrame(summary_records)
summary_csv = OUT_DIR / "task2_latency.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"Saved summary latency table to {summary_csv}")

print("\n" + "="*80)
print("TASK 2 LATENCY BENCHMARK RESULTS (Unrounded)")
print("="*80)
for _, r in df_summary.iterrows():
    print(f"{r['stage']}")
    print(f"  Median: {r['median_ms']:.6f} ms | IQR: {r['iqr_ms']:.6f} ms (Q1={r['q25_ms']:.6f}, Q3={r['q75_ms']:.6f})")
    print(f"  Mean:   {r['mean_ms']:.6f} ms +- {r['std_ms']:.6f} ms | Min: {r['min_ms']:.6f} ms, Max: {r['max_ms']:.6f} ms")
    print(f"  FPS:    {r['fps_at_median']:.2f}")

