import sys
from pathlib import Path
import torch
import torchvision
import numpy as np
import pandas as pd
from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from fa_evyolo_modules import SpectralAmplitudeNorm2d, EvidentialRegularizer, ReflectionUncertaintyNMS
from eval_table2_cleaned import preprocess_image_and_gt

setattr(ul_modules, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
setattr(ul_tasks, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_pt = Path(r"D:\Journal-Project\runs\detect\runs\Baseline_YOLO11n\train_marine_litter\weights\best.pt")
fa_pt = Path(r"D:\Journal-Project\runs\detect\runs\FA_EvYOLO\train_marine_litter_fa\weights\best.pt")

base_model = YOLO(str(base_pt))
fa_model = YOLO(str(fa_pt))

base_net = base_model.model.to(device).eval()
fa_net = fa_model.model.to(device).eval()

cv2_buffers, cv3_buffers = [], []
detect_head = fa_net.model[-1]
for s in range(3):
    detect_head.cv2[s].register_forward_hook(lambda m, inp, out: cv2_buffers.append(out))
    detect_head.cv3[s].register_forward_hook(lambda m, inp, out: cv3_buffers.append(out))

ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65).to(device)
benchmark_root = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark")
datasets = ["marine_litter", "floating_waste", "flow_img"]

def match_boxes(pred_boxes, gt_boxes, iou_thr=0.50):
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return matched_pred, matched_gt
    
    ious = torchvision.ops.box_iou(pred_boxes, gt_boxes).cpu().numpy() # (n_pred, n_gt)
    # Greedy matching sorted by IoU
    pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            if ious[i, j] >= iou_thr:
                pairs.append((ious[i, j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    for _, i, j in pairs:
        if not matched_pred[i] and not matched_gt[j]:
            matched_pred[i] = True
            matched_gt[j] = True
            
    return matched_pred, matched_gt

results = []

for ds in datasets:
    img_dir = benchmark_root / ds / "test" / "images"
    lbl_dir = benchmark_root / ds / "test" / "labels"
    img_files = sorted(list(img_dir.glob("*.*")))
    
    # Store per-model stats: TP, FN, FP, Glare_FP, FP_empty, Total_GT, Total_Empty_Imgs
    configs = ["M1_Baseline_YOLO11n", "M2_FA_EvYOLO_SAN_PCG", "M4_FA_EvYOLO_Full_RU_NMS"]
    stats = {cfg: {"tp": 0, "fn": 0, "fp": 0, "glare_fp": 0, "fp_empty": 0, "total_gt": 0, "empty_imgs": 0} for cfg in configs}
    
    with torch.no_grad():
        for p in img_files:
            l_p = lbl_dir / f"{p.stem}.txt"
            img, gt = preprocess_image_and_gt(p, l_p)
            if img is None: continue
            
            n_gt = gt.shape[0]
            is_empty_img = (n_gt == 0)
            for cfg in configs:
                stats[cfg]["total_gt"] += n_gt
                if is_empty_img:
                    stats[cfg]["empty_imgs"] += 1
            
            # --- M1: Baseline YOLO11n ---
            preds_base = base_net(img.unsqueeze(0))[0][0]
            scores_b = preds_base[4]
            mask_b = scores_b >= 0.15
            if mask_b.any():
                cx, cy = preds_base[0, mask_b], preds_base[1, mask_b]
                bw, bh = preds_base[2, mask_b], preds_base[3, mask_b]
                boxes_b = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
                sc_b = scores_b[mask_b]
                keep_b = torchvision.ops.nms(boxes_b, sc_b, 0.65)
                std_boxes_b = boxes_b[keep_b]
                glare_d_b = ru_nms.compute_box_glare_density(img, std_boxes_b)
                m_pred_b, m_gt_b = match_boxes(std_boxes_b, gt, 0.50)
                
                tp_b = int(np.sum(m_pred_b))
                fp_b = int(np.sum(~m_pred_b))
                fn_b = n_gt - int(np.sum(m_gt_b))
                gl_fp_b = int(np.sum((~m_pred_b) & (glare_d_b.cpu().numpy() >= 0.20)))
                
                stats["M1_Baseline_YOLO11n"]["tp"] += tp_b
                stats["M1_Baseline_YOLO11n"]["fp"] += fp_b
                stats["M1_Baseline_YOLO11n"]["fn"] += fn_b
                stats["M1_Baseline_YOLO11n"]["glare_fp"] += gl_fp_b
                if is_empty_img:
                    stats["M1_Baseline_YOLO11n"]["fp_empty"] += fp_b
            else:
                stats["M1_Baseline_YOLO11n"]["fn"] += n_gt
                
            # --- FA-EvYOLO Forward (for M2 and M4) ---
            cv2_buffers.clear()
            cv3_buffers.clear()
            preds_fa = fa_net(img.unsqueeze(0))[0][0]
            scores_fa = preds_fa[4]
            mask_fa = scores_fa >= 0.15
            
            if mask_fa.any():
                dfl_flat = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
                cls_flat = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
                
                cx, cy = preds_fa[0, mask_fa], preds_fa[1, mask_fa]
                bw, bh = preds_fa[2, mask_fa], preds_fa[3, mask_fa]
                boxes_fa = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
                cand_scores = scores_fa[mask_fa]
                cand_cls = cls_flat[mask_fa, 0]
                cand_dfl = dfl_flat[mask_fa]
                
                unc = EvidentialRegularizer.extract_box_uncertainty(cand_cls, cand_dfl, reg_max=16)
                u_tot = unc["u_total"]
                
                # --- M2: FA-EvYOLO + Standard NMS ---
                keep_m2 = torchvision.ops.nms(boxes_fa, cand_scores, 0.65)
                boxes_m2 = boxes_fa[keep_m2]
                glare_d_m2 = ru_nms.compute_box_glare_density(img, boxes_m2)
                m_pred_m2, m_gt_m2 = match_boxes(boxes_m2, gt, 0.50)
                
                tp_m2 = int(np.sum(m_pred_m2))
                fp_m2 = int(np.sum(~m_pred_m2))
                fn_m2 = n_gt - int(np.sum(m_gt_m2))
                gl_fp_m2 = int(np.sum((~m_pred_m2) & (glare_d_m2.cpu().numpy() >= 0.20)))
                
                stats["M2_FA_EvYOLO_SAN_PCG"]["tp"] += tp_m2
                stats["M2_FA_EvYOLO_SAN_PCG"]["fp"] += fp_m2
                stats["M2_FA_EvYOLO_SAN_PCG"]["fn"] += fn_m2
                stats["M2_FA_EvYOLO_SAN_PCG"]["glare_fp"] += gl_fp_m2
                if is_empty_img:
                    stats["M2_FA_EvYOLO_SAN_PCG"]["fp_empty"] += fp_m2
                    
                # --- M4: FA-EvYOLO + Full RU-NMS ---
                boxes_m4, sc_m4, keep_m4 = ru_nms.modulate_and_filter(
                    boxes_fa, cand_scores, u_tot, img, conf_thres=0.15, use_glare=True, use_vacuity=True
                )
                if boxes_m4.shape[0] > 0:
                    glare_d_m4 = ru_nms.compute_box_glare_density(img, boxes_m4)
                    m_pred_m4, m_gt_m4 = match_boxes(boxes_m4, gt, 0.50)
                    
                    tp_m4 = int(np.sum(m_pred_m4))
                    fp_m4 = int(np.sum(~m_pred_m4))
                    fn_m4 = n_gt - int(np.sum(m_gt_m4))
                    gl_fp_m4 = int(np.sum((~m_pred_m4) & (glare_d_m4.cpu().numpy() >= 0.20)))
                    
                    stats["M4_FA_EvYOLO_Full_RU_NMS"]["tp"] += tp_m4
                    stats["M4_FA_EvYOLO_Full_RU_NMS"]["fp"] += fp_m4
                    stats["M4_FA_EvYOLO_Full_RU_NMS"]["fn"] += fn_m4
                    stats["M4_FA_EvYOLO_Full_RU_NMS"]["glare_fp"] += gl_fp_m4
                    if is_empty_img:
                        stats["M4_FA_EvYOLO_Full_RU_NMS"]["fp_empty"] += fp_m4
                else:
                    stats["M4_FA_EvYOLO_Full_RU_NMS"]["fn"] += n_gt
            else:
                stats["M2_FA_EvYOLO_SAN_PCG"]["fn"] += n_gt
                stats["M4_FA_EvYOLO_Full_RU_NMS"]["fn"] += n_gt

    for cfg in configs:
        st = stats[cfg]
        p = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else 0.0
        r = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results.append({
            "Dataset": ds,
            "Model": cfg,
            "Total_GT": st["total_gt"],
            "TP": st["tp"],
            "FN": st["fn"],
            "FP": st["fp"],
            "Glare_FP (rho>=0.20)": st["glare_fp"],
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "Empty_Images": st["empty_imgs"],
            "FP_on_Empty_Images": st["fp_empty"],
        })

df = pd.DataFrame(results)
out_csv = benchmark_root / "Table_A3_Operating_Point_Error_Analysis.csv"
df.to_csv(str(out_csv), index=False)
print("=== Table A3 Successfully Generated ===")
print(df.to_string(index=False))

