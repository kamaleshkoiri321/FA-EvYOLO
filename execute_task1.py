"""
execute_task1.py - Task 1: Settings Provenance
Documents the origin, date, dataset, and code/notes evidence for each hyperparameter.
Saves CSV to results_v6/final_audit/task1_settings_provenance.csv.
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path(r"D:\Journal-Project\results_v6\final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = [
    {
        "parameter": "gamma_u, gamma_g",
        "value": "0.35, 0.65",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "fa_evyolo_modules.py:190 (ReflectionUncertaintyNMS.__init__)",
        "notes_evidence": "Written in Cell 1.0 of phase1.ipynb as default evidential damping hyperparameters (s_mod = s * exp(-gamma_u*u - gamma_g*rho*u)). Later verified via post-hoc sensitivity grid in results_v6/A5/ru_nms_sensitivity_grid.csv across all three datasets."
    },
    {
        "parameter": "u weights (u_cls / u_loc)",
        "value": "0.55 / 0.45",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "fa_evyolo_modules.py:185 (EvidentialRegularizer.extract_box_uncertainty)",
        "notes_evidence": "Formula u_total = 0.55*u_cls + 0.45*u_loc defined in Cell 1.0 of phase1.ipynb. Hand-fixed to place slight majority weight on classification vacuity (u_cls) because glare false alarms primarily represent semantic background confusion rather than boundary jitter."
    },
    {
        "parameter": "u_loc weights (entropy / vacuity)",
        "value": "0.50 / 0.50",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "fa_evyolo_modules.py:181 (EvidentialRegularizer.extract_box_uncertainty)",
        "notes_evidence": "Formula u_loc = 0.50*norm_entropy + 0.50*dirichlet_vacuity defined in Cell 1.0 of phase1.ipynb. Combines distribution spread (Shannon entropy across 16 DFL bins) and evidential evidence vacuity in equal measure."
    },
    {
        "parameter": "glare map thresholds (c_max, c_max - c_min)",
        "value": "c_max >= 0.85, c_max - c_min <= 0.20",
        "data_source": "Fixed by hand before evaluation (physical optics)",
        "date_established": "2026-10-30",
        "code_reference": "fa_evyolo_modules.py:205-207 (ReflectionUncertaintyNMS.compute_box_glare_density)",
        "notes_evidence": "Defined in Cell 1.0 of phase1.ipynb based on Fresnel specular reflection physics on water: specular reflections are near-saturated (peak normalized RGB channel >= 0.85, ~217/255) and achromatic (low color saturation, max - min channel <= 0.20, ~51/255)."
    },
    {
        "parameter": "FP-G cutoff (glare false positive)",
        "value": "rho >= 0.20",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "results_v6/eval_canonical.py:25 (count_glare_fp) & eval_table2_cleaned.py",
        "notes_evidence": "Established as the standardized diagnostic threshold for attributing a false positive box to sun glint: a detection box is categorized as a glare false alarm if at least 20% of its bounding box area intersects specular glare mask pixels."
    },
    {
        "parameter": "operating point (conf, NMS IoU)",
        "value": "conf 0.15, IoU 0.65",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "results_v6/eval_canonical.py:45 & results_v6/A3/run_a3_canonical.py",
        "notes_evidence": "Standard robotic deployment operating point selected a priori to maximize recall on distant small debris (>75% COCO-small in all datasets) where default conf 0.25 suppresses low-contrast items; paired with Ultralytics default NMS IoU threshold 0.65."
    },
    {
        "parameter": "RSI weights & glint scale factor",
        "value": "0.50 / 0.30 / 0.20, factor 6 on f_glint",
        "data_source": "Fixed by hand before evaluation (benchmark stratification)",
        "date_established": "2026-10-30",
        "code_reference": "results_v6/run_c1.py:28 (compute_continuous_rsi) & phase1.ipynb:Cell 1.2",
        "notes_evidence": "Formula RSI = 0.50*min(1.0, 6.0*f_glint) + 0.30*e_streak + 0.20*contrast_tail created in Cell 1.2 of phase1.ipynb prior to training or evaluation to stratify datasets into reflection tiers. Factor 6 maps glint fraction to [0, 1] as specular highlights rarely exceed ~16.7% of total image surface."
    },
    {
        "parameter": "RSI thresholds (glint & streak mask)",
        "value": "V >= 220, S <= 50, V > 170",
        "data_source": "Fixed by hand before evaluation",
        "date_established": "2026-10-30",
        "code_reference": "results_v6/run_c1.py:17, 23 (compute_continuous_rsi) & phase1.ipynb:Cell 1.2",
        "notes_evidence": "Defined in Cell 1.2 of phase1.ipynb in HSV [0, 255] color space: V >= 220 and S <= 50 isolates white/bright sun glints; V > 170 defines bright water regions for directional Sobel gradient wave streak analysis. Fixed before building benchmark splits."
    }
]

df = pd.DataFrame(records)
csv_path = OUT_DIR / "task1_settings_provenance.csv"
df.to_csv(csv_path, index=False)
print(f"Task 1 complete. Saved {len(df)} rows to {csv_path}")
for _, r in df.iterrows():
    print(f"\n[-] Parameter: {r['parameter']} = {r['value']}")
    print(f"    Source:    {r['data_source']} ({r['date_established']})")
    print(f"    Code Ref:  {r['code_reference']}")
    print(f"    Evidence:  {r['notes_evidence']}")

