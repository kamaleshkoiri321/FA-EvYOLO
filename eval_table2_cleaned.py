"""
Cleaned Table 2 Evaluation Script for FA-EvYOLO
Implements:
1. Meaningful score threshold (conf_thres=0.15)
2. Isolated calibration on identical detection population (Standard NMS boxes)
3. End-to-End RU-NMS pipeline evaluation
4. Rigorous separation of True Positives, All False Positives, and Glare-Induced False Positives
5. Exact Mann-Whitney U statistical testing
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import torch
import torchvision

from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks

from fa_evyolo_modules import (
    SpectralAmplitudeNorm2d,
    EvidentialRegularizer,
    ReflectionUncertaintyNMS,
)

# Register custom module with ultralytics
setattr(ul_modules, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
setattr(ul_tasks, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_dece_and_brier(confidences: np.ndarray, is_tp: np.ndarray, n_bins: int = 10) -> tuple:
    if len(confidences) == 0:
        return 0.0, 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    dece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop = np.mean(mask)
        if prop > 0:
            acc = np.mean(is_tp[mask])
            conf = np.mean(confidences[mask])
            dece += abs(conf - acc) * prop
    brier = float(np.mean((confidences - is_tp) ** 2))
    return float(dece), float(brier)


def compute_auroc(y_true_error: np.ndarray, uncertainty_score: np.ndarray) -> float:
    y_true = np.asarray(y_true_error, dtype=np.float64)
    y_score = np.asarray(uncertainty_score, dtype=np.float64)
    n_pos = np.sum(y_true == 1.0)
    n_neg = np.sum(y_true == 0.0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    i, n = 0, len(y_score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    auc = (np.sum(ranks[y_true == 1.0]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_aupr(y_true_error: np.ndarray, uncertainty_score: np.ndarray) -> float:
    y_true = np.asarray(y_true_error, dtype=np.float64)
    y_score = np.asarray(uncertainty_score, dtype=np.float64)
    n_pos = np.sum(y_true == 1.0)
    if n_pos == 0:
        return 0.0
    desc = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[desc]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    prec = tps / np.maximum(tps + fps, 1e-12)
    rec = tps / n_pos
    rec_diff = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(prec * rec_diff))


def preprocess_image_and_gt(img_path: Path, label_path: Path, imgsz: int = 640):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    h0, w0 = img_bgr.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = (imgsz - new_unpad[0]) / 2.0, (imgsz - new_unpad[1]) / 2.0

    img_res = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR) if (w0, h0) != new_unpad else img_bgr
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_pad = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().to(device, dtype=torch.float32) / 255.0

    gt_boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * (w0 * r) + left
                y1 = (cy - bh / 2.0) * (h0 * r) + top
                x2 = (cx + bw / 2.0) * (w0 * r) + left
                y2 = (cy + bh / 2.0) * (h0 * r) + top
                gt_boxes.append([x1, y1, x2, y2])

    gt_xyxy = (
        torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        if gt_boxes
        else torch.zeros((0, 4), dtype=torch.float32, device=device)
    )
    return img_tensor, gt_xyxy


def match_tp_fp(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thr: float = 0.50) -> np.ndarray:
    n_pred = pred_boxes.shape[0]
    is_tp = np.zeros(n_pred, dtype=np.float64)
    if n_pred == 0 or gt_boxes.shape[0] == 0:
        return is_tp
    ious = torchvision.ops.box_iou(pred_boxes, gt_boxes).cpu().numpy()
    matched_gt = set()
    for i in range(n_pred):
        best_gt = -1
        best_iou = iou_thr
        for j in range(gt_boxes.shape[0]):
            if j not in matched_gt and ious[i, j] >= best_iou:
                best_iou = ious[i, j]
                best_gt = j
        if best_gt != -1:
            is_tp[i] = 1.0
            matched_gt.add(best_gt)
    return is_tp


def main():
    weights_path = Path(r"D:\Journal-Project\runs\detect\runs\FA_EvYOLO\train_marine_litter_fa\weights\best.pt")
    benchmark_root = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark")
    datasets = ["marine_litter", "floating_waste", "flow_img"]

    print(f"Loading weights from: {weights_path}")
    fa_yolo = YOLO("fa_evyolo.yaml")
    fa_yolo.load(str(weights_path))
    fa_net = fa_yolo.model.to(device).eval()

    detect_head = fa_net.model[-1]
    cv2_buffers, cv3_buffers = [], []

    for scale_idx in range(3):
        detect_head.cv2[scale_idx]._forward_hooks.clear()
        detect_head.cv3[scale_idx]._forward_hooks.clear()
        detect_head.cv2[scale_idx].register_forward_hook(lambda m, inp, out: cv2_buffers.append(out))
        detect_head.cv3[scale_idx].register_forward_hook(lambda m, inp, out: cv3_buffers.append(out))

    ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65)
    records = []

    for ds_name in datasets:
        print(f"\n--- Evaluating {ds_name.upper()} ---")
        img_dir = benchmark_root / ds_name / "test" / "images"
        lbl_dir = benchmark_root / ds_name / "test" / "labels"
        img_files = sorted(list(img_dir.glob("*.*")))

        # Metrics lists
        # Isolated calibration (evaluated on the EXACT SAME boxes kept by standard NMS)
        iso_raw_confs, iso_cal_confs = [], []
        iso_is_tp = []
        iso_u_cls, iso_u_joint, iso_glare = [], [], []

        # Full pipeline (RU-NMS with conf_thres=0.15)
        runms_confs, runms_is_tp = [], []
        runms_u_joint = []

        with torch.no_grad():
            for img_path in img_files:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                img_tensor, gt_xyxy = preprocess_image_and_gt(img_path, lbl_path)
                if img_tensor is None:
                    continue

                cv2_buffers.clear()
                cv3_buffers.clear()
                preds = fa_net(img_tensor.unsqueeze(0))
                pred_tensor = preds[0][0]

                dfl_flat = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
                cls_flat = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)

                scores = pred_tensor[4]
                cand_mask = scores >= 0.15
                if not cand_mask.any():
                    continue

                cx, cy = pred_tensor[0, cand_mask], pred_tensor[1, cand_mask]
                bw, bh = pred_tensor[2, cand_mask], pred_tensor[3, cand_mask]
                boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
                cand_scores = scores[cand_mask].to(torch.float32)
                cand_cls = cls_flat[cand_mask, 0]
                cand_dfl = dfl_flat[cand_mask]

                # Uncertainty extraction
                unc_dict = EvidentialRegularizer.extract_box_uncertainty(cand_cls, cand_dfl, reg_max=16)
                u_cls = unc_dict["u_cls"]
                u_total = unc_dict["u_total"]

                glare_density = ru_nms.compute_box_glare_density(img_tensor, boxes_xyxy)
                u_joint = u_total * (1.0 + 0.65 * glare_density)

                # ==========================================
                # 1. ISOLATED CALIBRATION (Same Boxes)
                # ==========================================
                keep_std = torchvision.ops.nms(boxes_xyxy, cand_scores, 0.65)
                std_boxes = boxes_xyxy[keep_std]
                std_raw_s = cand_scores[keep_std]
                std_u_tot = u_total[keep_std]
                std_glare = glare_density[keep_std]

                # Modulate scores on the EXACT SAME boxes
                damping = torch.exp(-0.35 * std_u_tot - 0.65 * std_glare * std_u_tot)
                std_cal_s = std_raw_s * damping

                tp_std = match_tp_fp(std_boxes, gt_xyxy, iou_thr=0.50)

                iso_raw_confs.append(std_raw_s.cpu().numpy())
                iso_cal_confs.append(std_cal_s.cpu().numpy())
                iso_is_tp.append(tp_std)
                iso_u_cls.append(u_cls[keep_std].cpu().numpy())
                iso_u_joint.append(u_joint[keep_std].cpu().numpy())
                iso_glare.append(std_glare.cpu().numpy())

                # ==========================================
                # 2. FULL RU-NMS PIPELINE (conf_thres=0.15)
                # ==========================================
                mod_boxes, mod_scores, keep_ru = ru_nms.modulate_and_filter(
                    boxes_xyxy, cand_scores, u_total, img_tensor, conf_thres=0.15
                )
                if mod_boxes.shape[0] > 0:
                    tp_ru = match_tp_fp(mod_boxes, gt_xyxy, iou_thr=0.50)
                    runms_confs.append(mod_scores.cpu().numpy())
                    runms_is_tp.append(tp_ru)
                    runms_u_joint.append(u_joint[keep_ru].cpu().numpy())

        # Concatenate isolated results
        confs_raw = np.concatenate(iso_raw_confs)
        confs_cal = np.concatenate(iso_cal_confs)
        is_tp = np.concatenate(iso_is_tp)
        is_fp = 1.0 - is_tp

        u_cls_vec = np.concatenate(iso_u_cls)
        u_joint_vec = np.concatenate(iso_u_joint)
        glare_vec = np.concatenate(iso_glare)

        # Isolated D-ECE and Brier
        dece_raw, brier_raw = compute_dece_and_brier(confs_raw, is_tp)
        dece_cal, brier_cal = compute_dece_and_brier(confs_cal, is_tp)

        # Error Detection AUROC
        auroc_u_cls = compute_auroc(is_fp, u_cls_vec)
        auroc_joint = compute_auroc(is_fp, u_joint_vec)
        aupr_joint = compute_aupr(is_fp, u_joint_vec)

        # False Positives Breakdown (All vs. Specular Glare vs. Non-Glare)
        tp_mask = is_tp == 1.0
        fp_mask = is_fp == 1.0
        glare_fp_mask = fp_mask & (glare_vec >= 0.20)
        nonglare_fp_mask = fp_mask & (glare_vec < 0.20)

        u_tp_mean = float(np.mean(u_joint_vec[tp_mask])) if np.any(tp_mask) else 0.0
        u_fp_all_mean = float(np.mean(u_joint_vec[fp_mask])) if np.any(fp_mask) else 0.0
        u_fp_glare_mean = float(np.mean(u_joint_vec[glare_fp_mask])) if np.any(glare_fp_mask) else 0.0
        u_fp_nonglare_mean = float(np.mean(u_joint_vec[nonglare_fp_mask])) if np.any(nonglare_fp_mask) else 0.0

        # Statistical separation test (Mann-Whitney U: FP vs TP)
        _, p_val_all = mannwhitneyu(u_joint_vec[fp_mask], u_joint_vec[tp_mask], alternative="greater")
        p_val_glare = (
            mannwhitneyu(u_joint_vec[glare_fp_mask], u_joint_vec[tp_mask], alternative="greater")[1]
            if np.any(glare_fp_mask)
            else 1.0
        )

        # End-to-End Pipeline Metrics
        if len(runms_confs) > 0:
            confs_ru = np.concatenate(runms_confs)
            tp_ru = np.concatenate(runms_is_tp)
            pipeline_dece, pipeline_brier = compute_dece_and_brier(confs_ru, tp_ru)
        else:
            pipeline_dece, pipeline_brier = 0.0, 0.0

        records.append({
            "Dataset": ds_name,
            "Domain": "In-Domain" if ds_name == "marine_litter" else "Zero-Shot OOD",
            "Detections_Evaluated": len(confs_raw),
            "Uncal_D-ECE": round(dece_raw, 4),
            "Cal_D-ECE (Same Box)": round(dece_cal, 4),
            "Pipeline_D-ECE (RU-NMS)": round(pipeline_dece, 4),
            "Uncal_Brier": round(brier_raw, 4),
            "Cal_Brier (Same Box)": round(brier_cal, 4),
            "AUROC_u_cls": round(auroc_u_cls, 4),
            "AUROC_Ev+Glare": round(auroc_joint, 4),
            "AUPR_Error_Detect": round(aupr_joint, 4),
            "Mean_u (TP)": round(u_tp_mean, 4),
            "Mean_u (FP_All)": round(u_fp_all_mean, 4),
            "Mean_u (FP_Glare)": round(u_fp_glare_mean, 4),
            "Mean_u (FP_NonGlare)": round(u_fp_nonglare_mean, 4),
            "MannWhitney_p (All)": f"{p_val_all:.2e}",
            "MannWhitney_p (Glare)": f"{p_val_glare:.2e}",
        })

    for scale_idx in range(3):
        detect_head.cv2[scale_idx]._forward_hooks.clear()
        detect_head.cv3[scale_idx]._forward_hooks.clear()

    df_cleaned = pd.DataFrame(records)
    out_csv = benchmark_root / "Table2_Uncertainty_Calibration_and_Pillar_Verification.csv"
    df_cleaned.to_csv(out_csv, index=False)

    print("\n=========================================================================")
    print("SUCCESS: Updated Table 2 with Cleaned Scientific Metrics Saved To:")
    print(out_csv)
    print("=========================================================================\n")
    print(df_cleaned.to_string(index=False))


if __name__ == "__main__":
    main()

