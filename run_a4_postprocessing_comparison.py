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

base_net = YOLO(str(base_pt)).model.to(device).eval()
fa_net = YOLO(str(fa_pt)).model.to(device).eval()

cv2_buffers, cv3_buffers = [], []
detect_head = fa_net.model[-1]
for s in range(3):
    detect_head.cv2[s].register_forward_hook(lambda m, inp, out: cv2_buffers.append(out))
    detect_head.cv3[s].register_forward_hook(lambda m, inp, out: cv3_buffers.append(out))

ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65).to(device)
benchmark_root = Path(r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark")
datasets = ["marine_litter", "floating_waste", "flow_img"]

def soft_nms_gaussian(boxes, scores, sigma=0.5, score_thr=0.15):
    if boxes.shape[0] == 0:
        return torch.zeros((0, 4), device=boxes.device), torch.zeros((0,), device=scores.device)
    b = boxes.clone()
    s = scores.clone()
    keep_boxes = []
    keep_scores = []
    
    while b.shape[0] > 0 and len(keep_boxes) < 300:
        max_idx = torch.argmax(s)
        max_box = b[max_idx]
        max_score = s[max_idx]
        
        if max_score < score_thr:
            break
            
        keep_boxes.append(max_box)
        keep_scores.append(max_score)
        
        b = torch.cat([b[:max_idx], b[max_idx+1:]], dim=0)
        s = torch.cat([s[:max_idx], s[max_idx+1:]], dim=0)
        
        if b.shape[0] == 0:
            break
            
        ious = torchvision.ops.box_iou(max_box.unsqueeze(0), b)[0]
        decay = torch.exp(-(ious ** 2) / sigma)
        s = s * decay
        valid = s >= 0.05
        b = b[valid]
        s = s[valid]
        
    if len(keep_boxes) == 0:
        return torch.zeros((0, 4), device=boxes.device), torch.zeros((0,), device=scores.device)
    return torch.stack(keep_boxes), torch.stack(keep_scores)

def compute_ap50(all_scores, all_tp, total_gt):
    if len(all_scores) == 0 or total_gt == 0:
        return 0.0
    scores = np.array(all_scores)
    tp = np.array(all_tp)
    sort_idx = np.argsort(-scores)
    tp_sorted = tp[sort_idx]
    fp_sorted = 1.0 - tp_sorted
    
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    
    prec = cum_tp / (cum_tp + cum_fp)
    rec = cum_tp / total_gt
    
    # 101-point COCO interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return float(ap)

# Define post-processing variants to evaluate:
# (a) standard NMS at conf 0.15;
# (b) standard NMS at conf 0.20, 0.25, 0.30;
# (c) Soft-NMS, Gaussian sigma = 0.5;
# (d) uncertainty-only damping (FA-EvYOLO only);
# (e) full RU-NMS (FA-EvYOLO only).

methods = [
    ("Std_NMS_conf0.15", 0.15, "std_nms"),
    ("Std_NMS_conf0.20", 0.20, "std_nms"),
    ("Std_NMS_conf0.25", 0.25, "std_nms"),
    ("Std_NMS_conf0.30", 0.30, "std_nms"),
    ("Soft_NMS_sigma0.5", 0.15, "soft_nms"),
    ("Uncertainty_Only_Damping", 0.15, "vacuity_nms"),
    ("Full_RU_NMS", 0.15, "full_ru_nms"),
]

results = []

for ds in datasets:
    img_dir = benchmark_root / ds / "test" / "images"
    lbl_dir = benchmark_root / ds / "test" / "labels"
    img_files = sorted(list(img_dir.glob("*.*")))
    
    # Pre-load predictions
    cache = []
    total_gt = 0
    with torch.no_grad():
        for p in img_files:
            l_p = lbl_dir / f"{p.stem}.txt"
            img, gt = preprocess_image_and_gt(p, l_p)
            if img is None: continue
            total_gt += gt.shape[0]
            
            # Base preds
            pb = base_net(img.unsqueeze(0))[0][0]
            
            # FA preds
            cv2_buffers.clear()
            cv3_buffers.clear()
            pfa = fa_net(img.unsqueeze(0))[0][0]
            dfl_f = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
            cls_f = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
            
            cache.append((img, gt, pb, pfa, cls_f, dfl_f))
            
    for net_name in ["Baseline_YOLO11n", "FA_EvYOLO"]:
        for m_name, thr, m_type in methods:
            if net_name == "Baseline_YOLO11n" and m_type in ["vacuity_nms", "full_ru_nms"]:
                continue # Baseline does not have evidential uncertainty
                
            all_scores, all_tp = [], []
            total_tp, total_fp, total_fn, total_glare_fp = 0, 0, 0, 0
            
            for img, gt, pb, pfa, cls_f, dfl_f in cache:
                n_gt = gt.shape[0]
                preds = pb if net_name == "Baseline_YOLO11n" else pfa
                scores = preds[4]
                mask = scores >= 0.05
                
                if not mask.any():
                    total_fn += n_gt
                    continue
                    
                cx, cy, bw, bh = preds[0, mask], preds[1, mask], preds[2, mask], preds[3, mask]
                boxes = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
                c_scores = scores[mask]
                
                if m_type == "std_nms":
                    m = c_scores >= thr
                    if not m.any():
                        total_fn += n_gt
                        continue
                    b_sub = boxes[m]
                    s_sub = c_scores[m]
                    keep = torchvision.ops.nms(b_sub, s_sub, 0.65)
                    det_boxes = b_sub[keep]
                    det_scores = s_sub[keep]
                elif m_type == "soft_nms":
                    det_boxes, det_scores = soft_nms_gaussian(boxes, c_scores, sigma=0.5, score_thr=thr)
                elif m_type == "vacuity_nms":
                    cand_cls = cls_f[mask, 0]
                    cand_dfl = dfl_f[mask]
                    unc = EvidentialRegularizer.extract_box_uncertainty(cand_cls, cand_dfl, reg_max=16)
                    det_boxes, det_scores, _ = ru_nms.modulate_and_filter(
                        boxes, c_scores, unc["u_total"], img, conf_thres=thr, use_glare=False, use_vacuity=True
                    )
                elif m_type == "full_ru_nms":
                    cand_cls = cls_f[mask, 0]
                    cand_dfl = dfl_f[mask]
                    unc = EvidentialRegularizer.extract_box_uncertainty(cand_cls, cand_dfl, reg_max=16)
                    det_boxes, det_scores, _ = ru_nms.modulate_and_filter(
                        boxes, c_scores, unc["u_total"], img, conf_thres=thr, use_glare=True, use_vacuity=True
                    )
                
                if det_boxes.shape[0] == 0:
                    total_fn += n_gt
                    continue
                    
                glare_d = ru_nms.compute_box_glare_density(img, det_boxes).cpu().numpy()
                
                # Match
                n_det = det_boxes.shape[0]
                matched_det = np.zeros(n_det, dtype=bool)
                matched_gt = np.zeros(n_gt, dtype=bool)
                
                if n_det > 0 and n_gt > 0:
                    ious = torchvision.ops.box_iou(det_boxes, gt).cpu().numpy()
                    pairs = []
                    for i in range(n_det):
                        for j in range(n_gt):
                            if ious[i, j] >= 0.50:
                                pairs.append((ious[i, j], i, j))
                    pairs.sort(reverse=True, key=lambda x: x[0])
                    for _, i, j in pairs:
                        if not matched_det[i] and not matched_gt[j]:
                            matched_det[i] = True
                            matched_gt[j] = True
                            
                tp = int(np.sum(matched_det))
                fp = int(np.sum(~matched_det))
                fn = n_gt - int(np.sum(matched_gt))
                gl_fp = int(np.sum((~matched_det) & (glare_d >= 0.20)))
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_glare_fp += gl_fp
                
                for s_val, is_m in zip(det_scores.cpu().numpy(), matched_det):
                    all_scores.append(float(s_val))
                    all_tp.append(1.0 if is_m else 0.0)
                    
            p_val = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
            ap50 = compute_ap50(all_scores, all_tp, total_gt)
            
            results.append({
                "Dataset": ds,
                "Model": net_name,
                "Method": m_name,
                "Precision": round(p_val, 4),
                "Recall": round(r_val, 4),
                "F1": round(f1_val, 4),
                "AP50": round(ap50, 4),
                "FP": total_fp,
                "Glare_FP": total_glare_fp,
            })

df = pd.DataFrame(results)
out_csv = benchmark_root / "Table_A4_PostProcessing_Comparison.csv"
df.to_csv(str(out_csv), index=False)
print("=== Table A4 Successfully Generated ===")
print(df.to_string(index=False))

