"""Custom PyTorch modules for the FA-EvYOLO architecture."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SpectralAmplitudeNorm2d(nn.Module):
    """
    Pillar 1B: Spectral Amplitude Normalization with Phase-Coherence Gate (SAN-PCG).
    Decouples domain-variant illumination/glare amplitude from structural phase
    in the 2D Fourier domain while maintaining FP32/FP16 compatibility under AMP.
    """

    def __init__(self, c1: int = 64, c2: int = None, eps: float = 1e-4) -> None:
        super().__init__()
        c_out = c1 if c2 is None else c2
        self.c = c_out
        self.eps = eps

        self.proj = nn.Identity() if c1 == c_out else nn.Conv2d(c1, c_out, kernel_size=1, bias=False)
        self.amp_gamma = nn.Parameter(torch.ones(1, c_out, 1, 1))
        self.amp_beta = nn.Parameter(torch.zeros(1, c_out, 1, 1))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        self.phase_gate = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.post_norm = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = self.proj(x)
        _, _, h, w = x.shape

        # Fourier transform operations require FP32 because cuFFT only supports
        # power-of-two dimensions for FP16, and feature maps (e.g. 80x80, 40x40, 20x20)
        # are not powers of two.
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            fft_x = torch.fft.rfft2(x_fp32, norm="ortho")
            real = fft_x.real
            imag = fft_x.imag

            amp = torch.sqrt(real * real + imag * imag + self.eps)
            phase_real = real / amp
            phase_imag = imag / amp
            phase_unit = torch.complex(phase_real, phase_imag)

            log_amp = torch.log(amp.clamp(min=self.eps))
            mean = log_amp.mean(dim=(-2, -1), keepdim=True)
            std = log_amp.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp(min=1e-2)
            z = ((log_amp - mean) / std).clamp(-4.0, 4.0)

            gamma = self.amp_gamma.float().clamp(-1.5, 1.5)
            beta = self.amp_beta.float().clamp(-1.5, 1.5)
            norm_amp = torch.exp(z * gamma + beta)

            alpha = torch.sigmoid(self.mix_logit.float())
            blended_amp = alpha * norm_amp + (1.0 - alpha) * amp

            fft_norm = torch.complex(blended_amp * phase_real, blended_amp * phase_imag)
            x_spec = torch.fft.irfft2(fft_norm, s=(h, w), norm="ortho")
            x_phase = torch.fft.irfft2(phase_unit, s=(h, w), norm="ortho")

        w_dtype = self.phase_gate[0].weight.dtype
        x_spec = x_spec.to(w_dtype)
        x_phase = x_phase.to(w_dtype)

        gate = self.phase_gate(x_phase)
        out = self.act(self.post_norm(x_spec * gate))

        alpha_act = torch.sigmoid(self.mix_logit).to(orig_dtype)
        res = x + alpha_act * out.to(orig_dtype)

        return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)


class RetinexFourierTransform(nn.Module):
    """
    Pillar 1A: Multi-Scale Retinex Reflectance Normalization & Fourier Amplitude Mix (Retinex-FAM).
    """

    def __init__(self, retinex_weight: float = 0.25, fam_prob: float = 0.50, fam_strength: float = 0.30) -> None:
        super().__init__()
        self.retinex_weight = retinex_weight
        self.fam_prob = fam_prob
        self.fam_strength = fam_strength

    @torch.no_grad()
    def forward(self, img: torch.Tensor, training: bool = False) -> torch.Tensor:
        orig_dtype = img.dtype
        x = img.to(torch.float32).clamp(0.0, 1.0)

        illum_fine = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        illum_coarse = F.avg_pool2d(x, kernel_size=15, stride=1, padding=7)
        illum = 0.5 * (illum_fine + illum_coarse)

        reflectance = torch.log(x + 1e-3) - torch.log(illum + 1e-3)
        r_min = reflectance.amin(dim=(-2, -1), keepdim=True)
        r_max = reflectance.amax(dim=(-2, -1), keepdim=True)
        reflectance_norm = (reflectance - r_min) / (r_max - r_min + 1e-5)

        x = (1.0 - self.retinex_weight) * x + self.retinex_weight * reflectance_norm

        if training and x.shape[0] > 1 and torch.rand(1, device=x.device).item() < self.fam_prob:
            b, c, h, w = x.shape
            fft_x = torch.fft.rfft2(x, norm="ortho")
            amp = torch.abs(fft_x)
            phase = torch.angle(fft_x)

            perm = torch.randperm(b, device=x.device)
            lam = torch.empty((b, 1, 1, 1), device=x.device).uniform_(0.0, self.fam_strength)
            mixed_amp = (1.0 - lam) * amp + lam * amp[perm]

            fft_mixed = torch.polar(mixed_amp, phase)
            x = torch.fft.irfft2(fft_mixed, s=(h, w), norm="ortho")

        return x.clamp(0.0, 1.0).to(orig_dtype)


class EvidentialRegularizer(nn.Module):
    """
    Pillar 2: Unified Evidential Subjective Logic Regularizer (Ev-CLS & Ev-DFL)
    and Single-Pass Epistemic/Aleatoric Uncertainty Extractor.
    """

    def __init__(self, reg_max: int = 16, kl_weight: float = 0.05) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.kl_weight = kl_weight

    def compute_ev_cls_penalty(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        pred_fp32 = pred_scores.to(torch.float32).clamp(-15.0, 15.0)
        target_fp32 = target_scores.to(torch.float32).clamp(0.0, 1.0)

        e_pos = F.softplus(pred_fp32)
        e_neg = F.softplus(-pred_fp32)
        alpha = e_pos + 1.0
        beta = e_neg + 1.0
        strength = alpha + beta

        p_mean = alpha / strength
        bayes_var = (alpha * beta) / ((strength ** 2) * (strength + 1.0))
        err_sq = (target_fp32 - p_mean) ** 2

        bg_mask = (1.0 - target_fp32).clamp(0.0, 1.0)
        kl_bg = bg_mask * (e_pos ** 2) / (strength + 1e-5)

        return (err_sq + bayes_var + self.kl_weight * kl_bg).mean()

    def compute_ev_dfl_penalty(
        self, pred_dist: torch.Tensor, target_bins: torch.Tensor, fg_mask: torch.Tensor
    ) -> torch.Tensor:
        if fg_mask.sum() == 0:
            return pred_dist.sum() * 0.0

        fg_dist = pred_dist[fg_mask].to(torch.float32).view(-1, self.reg_max).clamp(-15.0, 15.0)
        alpha = F.softplus(fg_dist) + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / strength

        bins = torch.arange(self.reg_max, device=fg_dist.device, dtype=torch.float32).view(1, -1)
        expected_coord = (probs * bins).sum(dim=-1)
        second_moment = (probs * (bins ** 2)).sum(dim=-1)
        intrinsic_var = (second_moment - expected_coord ** 2).clamp(min=0.0)
        dirichlet_var = intrinsic_var / (strength.squeeze(-1) + 1.0)

        t_flat = target_bins.to(torch.float32).view(-1).clamp(0.0, float(self.reg_max - 1) - 1e-3)
        coord_err = F.smooth_l1_loss(expected_coord, t_flat, reduction="none")

        return (0.15 * coord_err + 0.10 * dirichlet_var).mean()

    @staticmethod
    @torch.no_grad()
    def extract_box_uncertainty(cls_logits: torch.Tensor, dfl_logits: torch.Tensor, reg_max: int = 16) -> dict:
        cls_fp32 = cls_logits.to(torch.float32).clamp(-10.0, 10.0)
        e_pos = 2.0 * torch.exp(cls_fp32)
        u_cls = (2.0 / (e_pos + 2.0)).clamp(0.0, 1.0)

        dfl_reshaped = dfl_logits.to(torch.float32).view(*dfl_logits.shape[:-1], 4, reg_max)
        p_dfl = F.softmax(dfl_reshaped, dim=-1)
        ent_dfl = -(p_dfl * torch.log(p_dfl.clamp(min=1e-8))).sum(dim=-1) / float(np.log(float(reg_max)))

        alpha_loc = F.softplus(dfl_reshaped) + 1.0
        vac_dfl = (float(reg_max) / alpha_loc.sum(dim=-1)).clamp(0.0, 1.0)
        u_loc = (0.5 * ent_dfl.mean(dim=-1) + 0.5 * vac_dfl.mean(dim=-1)).clamp(0.0, 1.0)

        if u_cls.ndim > u_loc.ndim:
            u_cls = u_cls.squeeze(-1)

        u_total = (0.55 * u_cls + 0.45 * u_loc).clamp(0.0, 1.0)
        return {"u_cls": u_cls, "u_loc": u_loc, "u_total": u_total}


class ReflectionUncertaintyNMS(nn.Module):
    """
    Pillar 3: Reflection-Aware Uncertainty-Modulated Non-Maximum Suppression (RU-NMS).
    """

    def __init__(self, gamma_u: float = 0.35, gamma_g: float = 0.65, iou_thres: float = 0.65) -> None:
        super().__init__()
        self.gamma_u = gamma_u
        self.gamma_g = gamma_g
        self.iou_thres = iou_thres

    @staticmethod
    @torch.no_grad()
    def compute_box_glare_density(img_tensor: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,), device=boxes_xyxy.device, dtype=torch.float32)

        _, h, w = img_tensor.shape
        c_max = img_tensor.amax(dim=0)
        c_min = img_tensor.amin(dim=0)
        glare_map = ((c_max >= 0.85) & ((c_max - c_min) <= 0.20)).to(torch.float32)

        integral = F.pad(glare_map, (1, 0, 1, 0), mode="constant", value=0.0).cumsum(dim=0).cumsum(dim=1)

        x1 = boxes_xyxy[:, 0].clamp(0, w - 1).long()
        y1 = boxes_xyxy[:, 1].clamp(0, h - 1).long()
        x2 = torch.maximum(boxes_xyxy[:, 2].clamp(1, w).long(), x1 + 1)
        y2 = torch.maximum(boxes_xyxy[:, 3].clamp(1, h).long(), y1 + 1)

        area = ((x2 - x1) * (y2 - y1)).to(torch.float32).clamp(min=1.0)
        glare_sum = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return (glare_sum / area).clamp(0.0, 1.0)

    @torch.no_grad()
    def modulate_and_filter(
        self,
        boxes_xyxy: torch.Tensor,
        scores: torch.Tensor,
        vacuity: torch.Tensor,
        img_tensor: torch.Tensor,
        conf_thres: float = 0.0,
        use_glare: bool = True,
        use_vacuity: bool = True,
    ) -> tuple:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy, scores, torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)

        glare_density = (
            self.compute_box_glare_density(img_tensor, boxes_xyxy)
            if use_glare
            else torch.zeros_like(scores, dtype=torch.float32)
        )
        u_flat = vacuity.view(-1).clamp(0.0, 1.0) if use_vacuity else torch.zeros_like(scores, dtype=torch.float32)
        s_flat = scores.view(-1).clamp(0.0, 1.0)

        damping = torch.exp(-self.gamma_u * u_flat - self.gamma_g * glare_density * u_flat)
        mod_scores = (s_flat * damping).clamp(0.0, 1.0)

        valid_mask = mod_scores >= conf_thres
        if not valid_mask.any():
            return (
                torch.zeros((0, 4), device=boxes_xyxy.device, dtype=boxes_xyxy.dtype),
                torch.zeros((0,), device=scores.device, dtype=scores.dtype),
                torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device),
            )

        valid_idx = torch.where(valid_mask)[0]
        nms_keep = torchvision.ops.nms(boxes_xyxy[valid_idx], mod_scores[valid_idx], self.iou_thres)
        keep_idx = valid_idx[nms_keep]
        return boxes_xyxy[keep_idx], mod_scores[keep_idx], keep_idx
