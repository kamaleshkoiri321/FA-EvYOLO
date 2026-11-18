"""
execute_task5.py - Task 5: Run B Qualitative Renders (M2 Standard NMS)
Protocol:
- M2: FA-EvYOLO Run B (seed 42, remapped head) with Standard NMS (conf 0.15, IoU 0.65)
- Baseline: YOLO11n (seed 42, MuSGD) with Standard NMS (conf 0.15, IoU 0.65)
- Matching: Greedy IoU >= 0.50 descending score order.
- Style: Ground Truth Green (0, 255, 0), YOLO11n TP Orange (0, 140, 255),
         FA-EvYOLO TP Blue (255, 100, 0), all FPs Red (0, 0, 255).
- Output: 640x640 letterboxed images with boxes only, NO text.

Images:
  1. floating_waste 401_jpg.rf.4189032f55748a241a6d6b4febdd42e1.jpg
  2. floating_waste frame_000260_jpg.rf.65dbb521672d64be89075738de0dc293.jpg
  3. flow_img 000009_jpg.rf.47e4b4d4fb3adccb05bc4f1e0396dfd4.jpg
  4. flow_img 392_jpg.rf.3ddfbf23440dca5f70a7d8c521a8fd31.jpg
  5. flow_img 181_jpg.rf.1d353eeefee0b1ddcc288fedbf7bbf8c.jpg
  6. flow_img 000293_jpg.rf.eac6737c283c2d04e0641782bfcd7104.jpg
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.ops as tv_ops

sys.path.insert(0, r"D:\Journal-Project")
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from ultralytics import YOLO
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
RENDERS_DIR = OUT_DIR / "task5_run_b_qualitative_renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark")
BASE_WEIGHTS = Path(r"D:\Journal-Project\results_v6\A1\train_yolo11n_musgd\weights\best.pt")
FA_WEIGHTS   = Path(r"D:\Journal-Project\results_v6\A2\train_fa_evyolo_remapped_seed42\weights\best.pt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_IMAGES = [
    ("floating_waste", "401_jpg.rf.4189032f55748a241a6d6b4febdd42e1.jpg"),
    ("floating_waste", "frame_000260_jpg.rf.65dbb521672d64be89075738de0dc293.jpg"),
    ("flow_img",       "000009_jpg.rf.47e4b4d4fb3adccb05bc4f1e0396dfd4.jpg"),
    ("flow_img",       "392_jpg.rf.3ddfbf23440dca5f70a7d8c521a8fd31.jpg"),
    ("flow_img",       "181_jpg.rf.1d353eeefee0b1ddcc288fedbf7bbf8c.jpg"),
    ("flow_img",       "000293_jpg.rf.eac6737c283c2d04e0641782bfcd7104.jpg"),
]

# BGR Colors
COLOR_GREEN  = (0, 255, 0)     # Ground Truth
COLOR_ORANGE = (0, 140, 255)   # YOLO11n TP
COLOR_BLUE   = (255, 100, 0)   # FA-EvYOLO TP (Vibrant Blue/Cyan)
COLOR_RED    = (0, 0, 255)     # All False Positives
LINE_THICKNESS = 2

def letterbox_image(img_bgr, imgsz=640):
    h0, w0 = img_bgr.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (imgsz - new_w) / 2.0, (imgsz - new_h) / 2.0
    img_res = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (w0, h0) != (new_w, new_h) else img_bgr
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img_pad = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_pad, (r, left, top, w0, h0)

def load_gt_boxes_640(label_path, scale_info):
    r, left, top, w0, h0 = scale_info
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * (w0 * r) + left
                y1 = (cy - bh / 2.0) * (h0 * r) + top
                x2 = (cx + bw / 2.0) * (w0 * r) + left
                y2 = (cy + bh / 2.0) * (h0 * r) + top
                boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32, device=DEVICE) if boxes else torch.zeros((0, 4), dtype=torch.float32, device=DEVICE)

def greedy_match(boxes_xyxy, scores, gt_boxes, iou_thr=0.50):
    if boxes_xyxy.numel() == 0 or gt_boxes.numel() == 0:
        return np.zeros(boxes_xyxy.shape[0], dtype=bool)
    ious = tv_ops.box_iou(boxes_xyxy, gt_boxes) # (N_pred, N_gt)
    order = torch.argsort(scores, descending=True)
    is_tp = np.zeros(boxes_xyxy.shape[0], dtype=bool)
    matched_gt = set()
    for idx in order:
        best_gt, best_iou = -1, -1.0
        for g in range(gt_boxes.shape[0]):
            if g in matched_gt:
                continue
            iou_val = float(ious[idx, g])
            if iou_val >= iou_thr and iou_val > best_iou:
                best_iou = iou_val
                best_gt = g
        if best_gt >= 0:
            is_tp[int(idx)] = True
            matched_gt.add(best_gt)
    return is_tp

def draw_boxes_clean(canvas, boxes, is_tp, tp_color, fp_color=COLOR_RED):
    vis = canvas.copy()
    if boxes.numel() > 0:
        b_np = boxes.cpu().numpy()
        for i in range(b_np.shape[0]):
            x1, y1, x2, y2 = map(int, np.round(b_np[i]))
            col = tp_color if is_tp[i] else fp_color
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, LINE_THICKNESS)
    return vis

def draw_gt_clean(canvas, gt_boxes, color=COLOR_GREEN):
    vis = canvas.copy()
    if gt_boxes.numel() > 0:
        for b in gt_boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, np.round(b))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, LINE_THICKNESS)
    return vis

def main():
    print(f"Device: {DEVICE}")
    base_model = YOLO(str(BASE_WEIGHTS))
    fa_model   = YOLO(str(FA_WEIGHTS))
    base_model.model.fuse()
    fa_model.model.fuse()
    base_net = base_model.model.to(DEVICE).eval()
    fa_net   = fa_model.model.to(DEVICE).eval()

    records = []

    for ds, fn in TARGET_IMAGES:
        stem = Path(fn).stem
        img_path = BENCHMARK / ds / "test" / "images" / fn
        lbl_path = BENCHMARK / ds / "test" / "labels" / (stem + ".txt")

        img_bgr = cv2.imread(str(img_path))
        assert img_bgr is not None, f"Could not read {img_path}"
        img_640, scale_info = letterbox_image(img_bgr, imgsz=640)
        gt_boxes = load_gt_boxes_640(lbl_path, scale_info)

        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0

        with torch.no_grad():
            b_preds = base_net(tensor)[0][0]
            fa_preds = fa_net(tensor)[0][0]

        # Decode & NMS M1 (Baseline YOLO11n: conf 0.15, IoU 0.65)
        b_scores = b_preds[4]
        b_mask = b_scores >= 0.15
        if b_mask.any():
            cx, cy, bw, bh = b_preds[0, b_mask], b_preds[1, b_mask], b_preds[2, b_mask], b_preds[3, b_mask]
            b_boxes = torch.stack([cx - 0.5*bw, cy - 0.5*bh, cx + 0.5*bw, cy + 0.5*bh], dim=-1)
            b_sc = b_scores[b_mask]
            keep_b = tv_ops.nms(b_boxes, b_sc, 0.65)
            m1_boxes, m1_scores = b_boxes[keep_b], b_sc[keep_b]
        else:
            m1_boxes, m1_scores = torch.zeros((0, 4), device=DEVICE), torch.zeros(0, device=DEVICE)

        # Decode & NMS M2 (FA-EvYOLO Standard NMS: conf 0.15, IoU 0.65)
        fa_scores = fa_preds[4]
        fa_mask = fa_scores >= 0.15
        if fa_mask.any():
            cx, cy, bw, bh = fa_preds[0, fa_mask], fa_preds[1, fa_mask], fa_preds[2, fa_mask], fa_preds[3, fa_mask]
            fa_b = torch.stack([cx - 0.5*bw, cy - 0.5*bh, cx + 0.5*bw, cy + 0.5*bh], dim=-1)
            fa_sc = fa_scores[fa_mask]
            keep_fa = tv_ops.nms(fa_b, fa_sc, 0.65)
            m2_boxes, m2_scores = fa_b[keep_fa], fa_sc[keep_fa]
        else:
            m2_boxes, m2_scores = torch.zeros((0, 4), device=DEVICE), torch.zeros(0, device=DEVICE)

        # Match to GT
        m1_tp = greedy_match(m1_boxes, m1_scores, gt_boxes, iou_thr=0.50)
        m2_tp = greedy_match(m2_boxes, m2_scores, gt_boxes, iou_thr=0.50)

        n_gt = int(gt_boxes.shape[0])
        n_m1_tp = int(np.sum(m1_tp))
        n_m1_fp = int(m1_boxes.shape[0] - n_m1_tp)
        n_m2_tp = int(np.sum(m2_tp))
        n_m2_fp = int(m2_boxes.shape[0] - n_m2_tp)

        # Render 3 versions (640x640, boxes only, no text)
        v1_path = RENDERS_DIR / f"{ds}_{stem}_1_gt.jpg"
        v2_path = RENDERS_DIR / f"{ds}_{stem}_2_yolo11n.jpg"
        v3_path = RENDERS_DIR / f"{ds}_{stem}_3_fa_evyolo.jpg"

        vis_gt = draw_gt_clean(img_640, gt_boxes, color=COLOR_GREEN)
        vis_m1 = draw_boxes_clean(img_640, m1_boxes, m1_tp, tp_color=COLOR_ORANGE, fp_color=COLOR_RED)
        vis_m2 = draw_boxes_clean(img_640, m2_boxes, m2_tp, tp_color=COLOR_BLUE, fp_color=COLOR_RED)

        cv2.imwrite(str(v1_path), vis_gt)
        cv2.imwrite(str(v2_path), vis_m1)
        cv2.imwrite(str(v3_path), vis_m2)

        records.append({
            "dataset": ds,
            "filename": fn,
            "gt_count": n_gt,
            "yolo11n_boxes": int(m1_boxes.shape[0]),
            "yolo11n_tp": n_m1_tp,
            "yolo11n_fp": n_m1_fp,
            "fa_evyolo_m2_boxes": int(m2_boxes.shape[0]),
            "fa_evyolo_m2_tp": n_m2_tp,
            "fa_evyolo_m2_fp": n_m2_fp,
            "gt_file": v1_path.name,
            "yolo11n_file": v2_path.name,
            "fa_evyolo_file": v3_path.name,
        })
        print(f"Rendered {ds} - {stem}: GT={n_gt} | YOLO11n: TP={n_m1_tp}, FP={n_m1_fp} | FA-EvYOLO M2: TP={n_m2_tp}, FP={n_m2_fp}")

    df = pd.DataFrame(records)
    csv_path = OUT_DIR / "task5_qualitative_detections.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Task 5 detection records to {csv_path}")

if __name__ == "__main__":
    main()

