# FA-EvYOLO

**Frequency-Aware Evidential YOLO for Glare-Robust Floating Litter Detection**

FA-EvYOLO extends YOLO11n to detect floating litter on water surfaces when there is strong sun glare and reflection. Plain detectors often mistake bright specular highlights for debris. FA-EvYOLO tackles this with three components ("pillars"):

1. **Spectral decoupling:** separates illumination and glare from object structure in the Fourier domain.
2. **Evidential uncertainty:** produces a per-box uncertainty estimate from a single forward pass.
3. **Reflection-aware NMS:** uses that uncertainty, together with local glare density, to suppress false detections caused by glare.

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Better zero-shot transfer:** trained on one dataset and tested on two unseen ones, FA-EvYOLO raises mAP@50 from 0.399 to 0.462 on FloW-Img and from 0.227 to 0.246 on Floating Waste.
- **Fewer false positives:** on Floating Waste, total false positives drop from 320 to 219 (−32%).
- **Uncertainty you can use:** per-box uncertainty separates true positives from errors with AUROC 0.77–0.89, and glare-induced false positives get the highest uncertainty.
- **Single pass:** uncertainty comes from one forward pass, with no MC-Dropout or ensembles.
- **Small changes to YOLO:** the code is built on Ultralytics YOLO11 and the new modules plug in through a custom model YAML.

---

## Method Overview

| Pillar | Module | Class in [`fa_evyolo_modules.py`](fa_evyolo_modules.py) | Role |
|---|---|---|---|
| 1A | **Retinex-FAM** | `RetinexFourierTransform` | Multi-scale Retinex reflectance normalisation, plus Fourier Amplitude Mixing as a training-time augmentation that simulates illumination shift |
| 1B | **SAN-PCG** | `SpectralAmplitudeNorm2d` | Spectral Amplitude Normalisation with a Phase-Coherence Gate. Placed before each of the three detection heads (P3/P4/P5), it normalises the domain-variant Fourier amplitude and keeps the structural phase |
| 2 | **Ev-CLS / Ev-DFL** | `EvidentialRegularizer` | Subjective-logic (Beta/Dirichlet) evidential losses on the classification and distribution-focal-loss branches. Gives classification uncertainty `u_cls`, localisation uncertainty `u_loc` and a combined `u_total` per box |
| 3 | **RU-NMS** | `ReflectionUncertaintyNMS` | Reflection-aware, uncertainty-modulated NMS. Lowers the score of each box according to its evidential vacuity and the glare density inside it before suppression |

The network is defined in [`fa_evyolo.yaml`](fa_evyolo.yaml). It is the YOLO11n backbone and neck, with three `SpectralAmplitudeNorm2d` layers placed in front of a single-class `Detect` head.

---

## Results

All models are trained on **Floating Marine Litter** only. Floating Waste and FloW-Img are evaluated **zero-shot** (no fine-tuning).

### Cross-dataset detection (all glare levels)

| Test set | Domain | YOLO11n mAP@50 | **FA-EvYOLO mAP@50** | YOLO11n mAP@50:95 | **FA-EvYOLO mAP@50:95** |
|---|---|---|---|---|---|
| Marine Litter | In-domain | 0.9245 | **0.9248** | **0.5929** | 0.5836 |
| Floating Waste | Zero-shot OOD | 0.2266 | **0.2462** | **0.0740** | 0.0733 |
| FloW-Img | Zero-shot OOD | 0.3987 | **0.4622** | 0.1468 | **0.1779** |

Results by glare level (Low / Medium / High Reflection Severity Index, RSI) are in [`Table1_CrossDataset_RSI_Matrix.csv`](Datasets/Q1_Harmonized_Benchmark/Table1_CrossDataset_RSI_Matrix.csv).

### Ablation

| Dataset | Configuration | Precision | Recall | mAP@50 | mAP@50:95 | Total FP |
|---|---|---|---|---|---|---|
| Floating Waste | M1 Baseline YOLO11n | 0.423 | 0.263 | 0.222 | 0.075 | 320 |
| | M2 + SAN-PCG | 0.528 | 0.270 | 0.244 | 0.078 | 216 |
| | M3 + Vacuity NMS | **0.530** | 0.269 | **0.247** | **0.079** | **213** |
| | M4 + Full RU-NMS | 0.524 | 0.269 | 0.240 | 0.076 | 219 |
| FloW-Img | M1 Baseline YOLO11n | 0.618 | 0.370 | 0.400 | 0.150 | 354 |
| | M2 + SAN-PCG | 0.658 | **0.422** | **0.461** | 0.182 | 339 |
| | M3 + Vacuity NMS | 0.660 | **0.422** | **0.461** | **0.183** | 336 |
| | M4 + Full RU-NMS | **0.665** | 0.419 | 0.457 | 0.179 | **327** |

The full matrix, including in-domain results, is in [`Table3_Comprehensive_Ablation_Matrix.csv`](Datasets/Q1_Harmonized_Benchmark/Table3_Comprehensive_Ablation_Matrix.csv).

### Uncertainty quality

| Dataset | AUROC (`u_cls`) | Mean *u* (TP) | Mean *u* (FP, glare) | Mean *u* (FP, non-glare) |
|---|---|---|---|---|
| Marine Litter | 0.887 | 0.353 | 0.646 | 0.526 |
| Floating Waste | 0.773 | 0.435 | 0.624 | 0.529 |
| FloW-Img | 0.815 | 0.453 | 0.676 | 0.566 |

On every dataset, the difference in uncertainty between true positives and false positives is statistically significant (Mann–Whitney *p* < 1e-18). Calibration metrics (D-ECE, Brier score) are in [`Table2_Uncertainty_Calibration_and_Pillar_Verification.csv`](Datasets/Q1_Harmonized_Benchmark/Table2_Uncertainty_Calibration_and_Pillar_Verification.csv).

---

## Installation

Tested with Python 3.10+, PyTorch 2.6 (CUDA 12.4) and Ultralytics 8.4.

```bash
git clone https://github.com/<your-username>/FA-EvYOLO.git
cd FA-EvYOLO

python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.182 opencv-python matplotlib pandas scipy imagehash
```

> **Note:** several scripts hard-code the project root as `D:\Journal-Project`. Change these paths, or the `path:` field in the dataset YAMLs, to match where you cloned the repository.

---

## Datasets

Three public Roboflow datasets (CC BY 4.0) are used:

| Dataset | Images | Original classes | Role |
|---|---|---|---|
| [Floating Marine Litter v3](https://universe.roboflow.com/sattwik-kumar-sahu-21241/floating-marine-litter/dataset/3) | 9,010 | metal_can, plastic_bag, plastic_bottle | Training + in-domain test |
| [Floating Waste v10](https://universe.roboflow.com/garbagedetection-gqqya/floating-waste/dataset/10) | 7,644 | bottle, can, carton, paper, plastic | Zero-shot OOD test |
| [FloW-Img v1](https://universe.roboflow.com/geetika-diw6y/flow-img-rnwvf/dataset/1) | 5,992 | bottle | Zero-shot OOD test |

For cross-dataset evaluation, all classes are merged into a single class, **`floating_litter`**. The test images are also split into Low, Medium and High strata by **Reflection Severity Index (RSI)**, a measure of how much glare an image contains. Near-duplicate images across splits were checked with perceptual hashing (pHash).

Download the datasets in **YOLO** format from the links above and place them under `Datasets/`. Then run cells 1.1–1.3 of [`phase1.ipynb`](phase1.ipynb). They clean the labels, merge the classes, split FloW-Img 80/10/10, compute RSI for every image and write the harmonised benchmark:

```
Datasets/Q1_Harmonized_Benchmark/
├── marine_litter/   {train,val,test}/  marine_litter.yaml  marine_litter_test_{Low,Med,High}_RSI.yaml
├── floating_waste/
└── flow_img/
```

---

## Training

The custom layer has to be registered with Ultralytics before the model YAML is parsed:

```python
from ultralytics import YOLO
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks
from fa_evyolo_modules import SpectralAmplitudeNorm2d

setattr(ul_modules, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)
setattr(ul_tasks, "SpectralAmplitudeNorm2d", SpectralAmplitudeNorm2d)

model = YOLO("fa_evyolo.yaml")
model.train(
    data="Datasets/Q1_Harmonized_Benchmark/marine_litter/marine_litter.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="MuSGD",
    seed=42,
)
```

The paper reports runs with seeds **42, 0 and 1**. See [`train.ipynb`](train.ipynb) for the full multi-seed procedure and [`train_a1_baseline.py`](train_a1_baseline.py) for the YOLO11n baseline.

---

## Evaluation

```bash
python journal_eval_a1.py                    # Per-image evaluation matching the Ultralytics validator
python journal_eval_a2.py                    # Paired bootstrap CIs: YOLO11n vs FA-EvYOLO
python journal_eval_a3.py                    # Brightness-tertile (dark / normal / bright) analysis
python eval_table2_cleaned.py                # Uncertainty calibration and pillar verification
python run_a3_error_table.py                 # Operating-point error analysis
python run_a4_postprocessing_comparison.py   # Standard NMS vs vacuity NMS vs RU-NMS
python benchmark_latency.py                  # Inference latency, baseline vs FA-EvYOLO
```

Supporting checks:

| Script | Purpose |
|---|---|
| `execute_task1.py` | Hyperparameter provenance |
| `execute_task2.py`, `execute_task2b.py` | Latency benchmark (single session; interleaved randomised timing) |
| `execute_task3.py`, `run_phash_check.py`, `compare_heldout_phash.py` | Perceptual-hash duplicate and overlap checks |
| `execute_task4.py` | AP confidence intervals via paired bootstrap |
| `execute_task5.py` | Qualitative renders |
| `execute_task8.py` | Illumination tertile split |

Results are written as CSV files to `results_v6/` and `Datasets/Q1_Harmonized_Benchmark/`.

---

## Inference

```bash
# Open a file dialog to choose an image
python predict.py

# Command-line usage
python predict.py --image path/to/image.jpg --conf 0.20 --iou 0.50 --no-show
```

| Flag | Default | Description |
|---|---|---|
| `--image` | – | Input image path (a file dialog opens if omitted) |
| `--conf` | `0.20` | Confidence threshold |
| `--iou` | `0.50` | IoU threshold for RU-NMS |
| `--tile` / `--no-tile` | auto | Force or disable high-resolution sliced tiling |
| `--no-baseline` | off | Skip the side-by-side comparison with YOLO11n |
| `--no-show` | off | Save the figure without opening a window |

By default, `predict.py` loads weights from `runs/detect/runs/FA_EvYOLO/train_marine_litter_fa/weights/best.pt`.

**Pretrained weights:** download them from the [Releases](../../releases) page and place them at the path above.

---

## Citation

If you use this code, please cite:

```bibtex
@article{faevyolo2026,
  title   = {FA-EvYOLO: Frequency-Aware Evidential YOLO for Glare-Robust Floating Litter Detection},
  author  = {<Author list>},
  journal = {<Journal>},
  year    = {2026}
}
```

---

## License

The code is released under the **AGPL-3.0** license, the same license as [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), on which it is built. The datasets keep their original **CC BY 4.0** licenses.

---

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the detection framework
- [Roboflow Universe](https://universe.roboflow.com) and the dataset authors for the Floating Marine Litter, Floating Waste and FloW-Img datasets
