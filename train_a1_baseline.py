import multiprocessing
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    multiprocessing.freeze_support()
    base_dir = Path(r"D:\Journal-Project\results_v6\A1")
    base_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = r"D:\Journal-Project\Datasets\Q1_Harmonized_Benchmark\marine_litter\marine_litter.yaml"

    print("--- Starting A1: Controlled Baseline Training (YOLO11n, MuSGD, lr0=0.01, seed=42) ---")
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="MuSGD",
        lr0=0.01,
        seed=42,
        deterministic=True,
        amp=True,
        workers=4,
        project=str(base_dir),
        name="train_yolo11n_musgd",
        exist_ok=True,
        val=False,
        verbose=True,
    )
    print("A1 Training completed successfully!")

if __name__ == "__main__":
    main()
