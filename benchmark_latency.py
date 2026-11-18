import time
import torch
import torchvision
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Benchmarking Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

base_pt = r"D:\Journal-Project\runs\detect\runs\Baseline_YOLO11n\train_marine_litter\weights\best.pt"
fa_pt = r"D:\Journal-Project\runs\detect\runs\FA_EvYOLO\train_marine_litter_fa\weights\best.pt"

# Load models and fuse
base_model = YOLO(base_pt)
fa_model = YOLO(fa_pt)

base_model.model.fuse()
fa_model.model.fuse()

base_net = base_model.model.to(device).eval()
fa_net = fa_model.model.to(device).eval()

dummy = torch.randn(1, 3, 640, 640, device=device)

base_params = sum(p.numel() for p in base_net.parameters())
fa_params = sum(p.numel() for p in fa_net.parameters())

from ultralytics.utils.torch_utils import get_flops
base_flops = get_flops(base_net, imgsz=640)
fa_flops = get_flops(fa_net, imgsz=640)

print(f"Baseline YOLO11n (Fused): Parameters = {base_params:,}, GFLOPs = {base_flops:.2f}")
print(f"FA-EvYOLO (Fused): Parameters = {fa_params:,}, GFLOPs = {fa_flops:.2f}")

n_iters = 300
conf_thres = 0.15

# 1. Baseline Latency (Forward Pass + Standard Conf Filtering + NMS)
with torch.no_grad():
    for _ in range(50):
        preds = base_net(dummy)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= conf_thres
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = torchvision.ops.nms(boxes, scores[mask], 0.65)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        preds = base_net(dummy)
        p = preds[0][0]
        scores = p[4]
        mask = scores >= conf_thres
        if mask.any():
            boxes = torch.stack([p[0, mask] - 0.5 * p[2, mask], p[1, mask] - 0.5 * p[3, mask],
                                 p[0, mask] + 0.5 * p[2, mask], p[1, mask] + 0.5 * p[3, mask]], dim=-1)
            _ = torchvision.ops.nms(boxes, scores[mask], 0.65)
    torch.cuda.synchronize()
    t_base = (time.perf_counter() - t0) / n_iters * 1000.0

print(f"\n1. Baseline YOLO11n (Forward + Conf Filtering + NMS): {t_base:.2f} ms ({1000.0/t_base:.1f} FPS)")

# 2. FA-EvYOLO Forward Only
with torch.no_grad():
    for _ in range(50):
        _ = fa_net(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = fa_net(dummy)
    torch.cuda.synchronize()
    t_fa_fwd = (time.perf_counter() - t0) / n_iters * 1000.0

print(f"2. FA-EvYOLO (Backbone+Neck+Head Forward Only): {t_fa_fwd:.2f} ms ({1000.0/t_fa_fwd:.1f} FPS)")

# 3. FA-EvYOLO Full Pipeline (Forward + Hooks + Evidential Uncertainty + Glare Density + RU-NMS)
cv2_buffers, cv3_buffers = [], []
detect_head = fa_net.model[-1]
for scale_idx in range(3):
    detect_head.cv2[scale_idx].register_forward_hook(lambda m, inp, out: cv2_buffers.append(out))
    detect_head.cv3[scale_idx].register_forward_hook(lambda m, inp, out: cv3_buffers.append(out))

ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=0.65).to(device)

with torch.no_grad():
    for _ in range(50):
        cv2_buffers.clear()
        cv3_buffers.clear()
        preds = fa_net(dummy)
        pred_tensor = preds[0][0]
        dfl_flat = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
        cls_flat = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
        scores = pred_tensor[4]
        cand_mask = scores >= conf_thres
        if cand_mask.any():
            cx, cy = pred_tensor[0, cand_mask], pred_tensor[1, cand_mask]
            bw, bh = pred_tensor[2, cand_mask], pred_tensor[3, cand_mask]
            boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
            cand_scores = scores[cand_mask]
            unc = EvidentialRegularizer.extract_box_uncertainty(cls_flat[cand_mask, 0], dfl_flat[cand_mask], reg_max=16)
            glare_d = ru_nms.compute_box_glare_density(dummy[0], boxes_xyxy)
            u_joint = unc["u_total"] * (1.0 + 0.65 * glare_d)
            mod_scores = cand_scores * (1.0 - 0.35 * unc["u_total"]) / (1.0 + 0.65 * glare_d)
            _ = torchvision.ops.nms(boxes_xyxy, mod_scores, 0.65)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        cv2_buffers.clear()
        cv3_buffers.clear()
        preds = fa_net(dummy)
        pred_tensor = preds[0][0]
        dfl_flat = torch.cat([t.flatten(2) for t in cv2_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
        cls_flat = torch.cat([t.flatten(2) for t in cv3_buffers], dim=2)[0].permute(1, 0).to(torch.float32)
        scores = pred_tensor[4]
        cand_mask = scores >= conf_thres
        if cand_mask.any():
            cx, cy = pred_tensor[0, cand_mask], pred_tensor[1, cand_mask]
            bw, bh = pred_tensor[2, cand_mask], pred_tensor[3, cand_mask]
            boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
            cand_scores = scores[cand_mask]
            unc = EvidentialRegularizer.extract_box_uncertainty(cls_flat[cand_mask, 0], dfl_flat[cand_mask], reg_max=16)
            glare_d = ru_nms.compute_box_glare_density(dummy[0], boxes_xyxy)
            u_joint = unc["u_total"] * (1.0 + 0.65 * glare_d)
            mod_scores = cand_scores * (1.0 - 0.35 * unc["u_total"]) / (1.0 + 0.65 * glare_d)
            _ = torchvision.ops.nms(boxes_xyxy, mod_scores, 0.65)
    torch.cuda.synchronize()
    t_fa_full = (time.perf_counter() - t0) / n_iters * 1000.0

print(f"3. FA-EvYOLO Full Pipeline (Forward + Hooks + Unc + RU-NMS): {t_fa_full:.2f} ms ({1000.0/t_fa_full:.1f} FPS)")
print(f"Full Pipeline Overhead vs Baseline: +{t_fa_full - t_base:.2f} ms")
