"""
Interactive & Command-Line Prediction Tool for FA-EvYOLO with RU-NMS
Supports Hybrid Full-Frame Letterbox + High-Resolution Sliced Tiling (SAHI)
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks

from fa_evyolo_modules import (
    SpectralAmplitudeNorm2d,
    ReflectionUncertaintyNMS,
    RetinexFourierTransform,
    EvidentialRegularizer,
)

setattr(ul_modules, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
setattr(ul_tasks, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
ul_tasks.__dict__["SpectralAmplitudeNorm2d"] = SpectralAmplitudeNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_image_dialog():
    """Opens a native Windows File Dialog to select an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select an Aquatic Image to Test with FA-EvYOLO",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[Warning] GUI dialog error: {e}")
        return None


def extract_patch_detections(p_tensor, fa_net, base_net, ru_nms, cv2_buf, cv3_buf, conf_thres):
    """Runs single-patch (640x640) inference for both Baseline and FA-EvYOLO."""
    b_boxes, b_scores = [], []
    if base_net is not None:
        with torch.no_grad():
            b_p = base_net(p_tensor.unsqueeze(0))[0][0]
            b_sc = b_p[4]
            b_mask = b_sc >= conf_thres
            if b_mask.any():
                cx, cy, bw, bh = b_p[0, b_mask], b_p[1, b_mask], b_p[2, b_mask], b_p[3, b_mask]
                b_boxes = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
                b_scores = b_sc[b_mask].to(torch.float32)

    cv2_buf.clear()
    cv3_buf.clear()
    fa_boxes, fa_scores, fa_vac, fa_glare = [], [], [], []
    with torch.no_grad():
        fa_p = fa_net(p_tensor.unsqueeze(0))[0][0]
        fa_sc = fa_p[4]
        fa_mask = fa_sc >= conf_thres
        if fa_mask.any():
            dfl_flat = torch.cat([t.flatten(2) for t in cv2_buf], dim=2)[0].permute(1, 0).to(torch.float32)
            cls_flat = torch.cat([t.flatten(2) for t in cv3_buf], dim=2)[0].permute(1, 0).to(torch.float32)

            cx, cy, bw, bh = fa_p[0, fa_mask], fa_p[1, fa_mask], fa_p[2, fa_mask], fa_p[3, fa_mask]
            fa_boxes = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
            fa_scores = fa_sc[fa_mask].to(torch.float32)
            cand_cls = cls_flat[fa_mask, 0]
            cand_dfl = dfl_flat[fa_mask]

            unc_dict = EvidentialRegularizer.extract_box_uncertainty(cand_cls, cand_dfl, reg_max=16)
            fa_vac = unc_dict["u_total"]
            fa_glare = ru_nms.compute_box_glare_density(p_tensor, fa_boxes)

    return (b_boxes, b_scores), (fa_boxes, fa_scores, fa_vac, fa_glare)


def run_hybrid_inference(img_bgr, fa_net, base_net, ru_nms, conf_thres=0.20, iou_thres=0.50, use_tiles=True):
    """
    Executes a Full-Frame 640x640 Letterbox Pass (mapped back to native w0 x h0 coordinates)
    and optionally fuses Overlapping 640x640 Tiles, followed by global RU-NMS.
    """
    h0, w0 = img_bgr.shape[:2]
    retinex = RetinexFourierTransform(retinex_weight=0.20).to(device)

    detect_head = fa_net.model[-1]
    cv2_buf, cv3_buf = [], []
    hk2 = [detect_head.cv2[i].register_forward_hook(lambda m, inp, o: cv2_buf.append(o)) for i in range(3)]
    hk3 = [detect_head.cv3[i].register_forward_hook(lambda m, inp, o: cv3_buf.append(o)) for i in range(3)]

    fa_all_boxes, fa_all_scores, fa_all_vac, fa_all_glare = [], [], [], []
    base_all_boxes, base_all_scores = [], []

    # 1. Global Full-Frame Letterbox Pass
    r = min(640.0 / h0, 640.0 / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = (640 - new_unpad[0]) / 2.0, (640 - new_unpad[1]) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_res = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_pad = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    p_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
    p_tensor = torch.from_numpy(p_rgb).permute(2, 0, 1).contiguous().to(device, dtype=torch.float32) / 255.0
    p_tensor_ret = retinex(p_tensor.unsqueeze(0), training=False)[0]

    (b_b, b_s), (f_b, f_s, f_v, f_g) = extract_patch_detections(
        p_tensor_ret, fa_net, base_net, ru_nms, cv2_buf, cv3_buf, conf_thres
    )
    if len(b_b) > 0:
        b_b_orig = b_b.clone()
        b_b_orig[:, [0, 2]] = ((b_b[:, [0, 2]] - left) / r).clamp(0, w0)
        b_b_orig[:, [1, 3]] = ((b_b[:, [1, 3]] - top) / r).clamp(0, h0)
        base_all_boxes.append(b_b_orig)
        base_all_scores.append(b_s)

    if len(f_b) > 0:
        f_b_orig = f_b.clone()
        f_b_orig[:, [0, 2]] = ((f_b[:, [0, 2]] - left) / r).clamp(0, w0)
        f_b_orig[:, [1, 3]] = ((f_b[:, [1, 3]] - top) / r).clamp(0, h0)
        fa_all_boxes.append(f_b_orig)
        fa_all_scores.append(f_s)
        fa_all_vac.append(f_v)
        fa_all_glare.append(f_g)

    # 2. High-Resolution Overlapping Tile Pass
    if use_tiles and (w0 > 640 or h0 > 640):
        tile_size = 640
        stride = int(tile_size * 0.70)
        for y in range(0, max(1, h0 - tile_size + stride), stride):
            for x in range(0, max(1, w0 - tile_size + stride), stride):
                x1 = min(x, max(0, w0 - tile_size))
                y1 = min(y, max(0, h0 - tile_size))
                x2 = min(x1 + tile_size, w0)
                y2 = min(y1 + tile_size, h0)

                patch = img_bgr[y1:y2, x1:x2]
                if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                    patch = cv2.copyMakeBorder(
                        patch,
                        0,
                        tile_size - patch.shape[0],
                        0,
                        tile_size - patch.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114),
                    )
                t_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                t_tensor = torch.from_numpy(t_rgb).permute(2, 0, 1).contiguous().to(device, dtype=torch.float32) / 255.0
                t_tensor_ret = retinex(t_tensor.unsqueeze(0), training=False)[0]

                (tb_b, tb_s), (tf_b, tf_s, tf_v, tf_g) = extract_patch_detections(
                    t_tensor_ret, fa_net, base_net, ru_nms, cv2_buf, cv3_buf, conf_thres
                )
                if len(tb_b) > 0:
                    tb_b_orig = tb_b.clone()
                    tb_b_orig[:, [0, 2]] = (tb_b[:, [0, 2]] + x1).clamp(0, w0)
                    tb_b_orig[:, [1, 3]] = (tb_b[:, [1, 3]] + y1).clamp(0, h0)
                    base_all_boxes.append(tb_b_orig)
                    base_all_scores.append(tb_s)

                if len(tf_b) > 0:
                    tf_b_orig = tf_b.clone()
                    tf_b_orig[:, [0, 2]] = (tf_b[:, [0, 2]] + x1).clamp(0, w0)
                    tf_b_orig[:, [1, 3]] = (tf_b[:, [1, 3]] + y1).clamp(0, h0)
                    fa_all_boxes.append(tf_b_orig)
                    fa_all_scores.append(tf_s)
                    fa_all_vac.append(tf_v)
                    fa_all_glare.append(tf_g)

    for h in hk2 + hk3:
        h.remove()

    # Global NMS for Baseline
    b_kept_boxes, b_kept_scores = np.zeros((0, 4)), np.zeros((0,))
    if len(base_all_boxes) > 0:
        b_tb = torch.cat(base_all_boxes, dim=0)
        b_ts = torch.cat(base_all_scores, dim=0)
        b_keep = torchvision.ops.nms(b_tb, b_ts, iou_thres)
        b_kept_boxes = b_tb[b_keep].cpu().numpy()
        b_kept_scores = b_ts[b_keep].cpu().numpy()

    # Global RU-NMS for FA-EvYOLO (with post-damping confidence filtering)
    fa_kept_boxes, fa_kept_scores, fa_kept_vac, fa_kept_glare = (
        np.zeros((0, 4)),
        np.zeros((0,)),
        np.zeros((0,)),
        np.zeros((0,)),
    )
    if len(fa_all_boxes) > 0:
        fa_tb = torch.cat(fa_all_boxes, dim=0)
        fa_ts = torch.cat(fa_all_scores, dim=0)
        fa_tv = torch.cat(fa_all_vac, dim=0)
        fa_tg = torch.cat(fa_all_glare, dim=0)

        damp = torch.exp(-ru_nms.gamma_u * fa_tv - ru_nms.gamma_g * fa_tg * fa_tv)
        fa_mod = (fa_ts * damp).clamp(0.0, 1.0)
        keep_mask = fa_mod >= conf_thres
        if keep_mask.any():
            fa_tb, fa_mod, fa_tv, fa_tg = fa_tb[keep_mask], fa_mod[keep_mask], fa_tv[keep_mask], fa_tg[keep_mask]
            fa_keep = torchvision.ops.nms(fa_tb, fa_mod, iou_thres)
            fa_kept_boxes = fa_tb[fa_keep].cpu().numpy()
            fa_kept_scores = fa_mod[fa_keep].cpu().numpy()
            fa_kept_vac = fa_tv[fa_keep].cpu().numpy()
            fa_kept_glare = fa_tg[fa_keep].cpu().numpy()

    return (b_kept_boxes, b_kept_scores), (fa_kept_boxes, fa_kept_scores, fa_kept_vac, fa_kept_glare)


def predict(
    image_path: str,
    conf_thres: float = 0.20,
    iou_thres: float = 0.50,
    show_baseline: bool = True,
    show_plot: bool = True,
    tile: bool = None,
):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[Error] Image file not found: {img_path}")
        return

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[Error] Could not read image at: {img_path}")
        return

    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if tile is None:
        tile = w0 >= 1000 or h0 >= 1000

    print("\n=======================================================")
    print(f"Analyzing: {img_path.name} ({w0}x{h0})")
    print(f"Mode: {'Hybrid Full-Frame + Sliced Tiling' if tile else 'Standard Full-Frame Letterbox'}")
    print(f"Confidence Threshold: {conf_thres} | IoU Threshold: {iou_thres} | Device: {device}")
    print("=======================================================")

    img_full_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().to(device, dtype=torch.float32) / 255.0
    c_max = img_full_tensor.amax(dim=0)
    c_min = img_full_tensor.amin(dim=0)
    glare_map = ((c_max >= 0.85) & ((c_max - c_min) <= 0.20)).float().cpu().numpy()
    glare_smoothed = cv2.GaussianBlur(glare_map, (45, 45), 0)
    overall_glare_pct = float(glare_map.mean() * 100.0)

    fa_weights = PROJECT_ROOT / "runs" / "detect" / "runs" / "FA_EvYOLO" / "train_marine_litter_fa" / "weights" / "best.pt"
    if not fa_weights.exists():
        fa_weights = PROJECT_ROOT / "runs" / "detect" / "runs" / "FA_EvYOLO" / "train_marine_litter_fa" / "weights" / "last.pt"
    fa_model = YOLO("fa_evyolo.yaml")
    fa_model.load(str(fa_weights))
    fa_net = fa_model.model.to(device).eval()

    base_net = None
    base_weights = PROJECT_ROOT / "runs" / "detect" / "runs" / "Baseline_YOLO11n" / "train_marine_litter" / "weights" / "best.pt"
    if show_baseline and base_weights.exists():
        base_model = YOLO(str(base_weights))
        base_net = base_model.model.to(device).eval()

    ru_nms = ReflectionUncertaintyNMS(gamma_u=0.35, gamma_g=0.65, iou_thres=iou_thres).to(device)

    (b_boxes, b_scores), (fa_boxes, fa_scores, fa_vacuity, fa_glare) = run_hybrid_inference(
        img_bgr, fa_net, base_net, ru_nms, conf_thres=conf_thres, iou_thres=iou_thres, use_tiles=tile
    )

    print("\n--- Detection Results ---")
    print(f"Overall Surface Glint: {overall_glare_pct:.2f}%")
    print(f"Baseline YOLO11n Detections: {len(b_boxes)}")
    print(f"FA-EvYOLO + RU-NMS Detections: {len(fa_boxes)}")
    for i, (b, s, u, g) in enumerate(zip(fa_boxes, fa_scores, fa_vacuity, fa_glare)):
        print(
            f"  Box {i+1}: Conf={s:.3f} | Vacuity(u)={u:.3f} | Box Glare={g*100:.1f}% | "
            f"Coordinates: [{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}]"
        )

    n_cols = 4 if show_baseline else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5), dpi=150)
    plt.subplots_adjust(wspace=0.08, top=0.88, bottom=0.05, left=0.02, right=0.98)

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"1. Input: {img_path.name}\n({w0}x{h0} Native)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].imshow(glare_smoothed, cmap="inferno", alpha=0.55)
    axes[1].set_title(f"2. Glare Map G(x,y)\nGlint: {overall_glare_pct:.1f}%", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    if show_baseline:
        axes[2].imshow(img_rgb)
        axes[2].set_title(f"3. Baseline YOLO11n\n({len(b_boxes)} detections)", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        for b, s in zip(b_boxes, b_scores):
            rect = patches.Rectangle(
                (b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=2.0, edgecolor="#ff1744", facecolor="none"
            )
            axes[2].add_patch(rect)
            axes[2].text(
                b[0],
                max(b[1] - 4, 15),
                f"Conf: {s:.3f}",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="#ff1744", alpha=0.85, pad=1.5, edgecolor="none"),
            )

    target_ax = axes[3] if show_baseline else axes[2]
    target_ax.imshow(img_rgb)
    target_ax.set_title(
        f"4. FA-EvYOLO + RU-NMS\n({len(fa_boxes)} calibrated detections)", fontsize=11, fontweight="bold"
    )
    target_ax.axis("off")
    for b, s, u in zip(fa_boxes, fa_scores, fa_vacuity):
        color = "#00e676" if u < 0.45 else "#ff9100"
        rect = patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=2.2, edgecolor=color, facecolor="none"
        )
        target_ax.add_patch(rect)
        label_text = f"Conf: {s:.3f} | u: {u:.2f}"
        target_ax.text(
            b[0],
            max(b[1] - 4, 15),
            label_text,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor="none"),
        )

    out_dir = PROJECT_ROOT / "runs" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_fa_evyolo_prediction.png"
    plt.savefig(str(out_path), bbox_inches="tight", dpi=200)
    print(f"\n[Success] Output figure saved to: {out_path}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test any aquatic image with FA-EvYOLO")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for RU-NMS (default: 0.50)")
    parser.add_argument("--tile", action="store_true", help="Force hybrid high-resolution slicing")
    parser.add_argument("--no-tile", action="store_true", help="Force standard full-frame letterbox only")
    parser.add_argument("--no-baseline", action="store_true", help="Do not compare against baseline YOLO11n")
    parser.add_argument("--no-show", action="store_true", help="Save figure without opening GUI window")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = select_image_dialog()
    if not image_path:
        print("[Exit] No image selected.")
        return

    tile_mode = True if args.tile else (False if args.no_tile else None)
    predict(
        image_path,
        conf_thres=args.conf,
        iou_thres=args.iou,
        show_baseline=not args.no_baseline,
        show_plot=not args.no_show,
        tile=tile_mode,
    )


if __name__ == "__main__":
    main()