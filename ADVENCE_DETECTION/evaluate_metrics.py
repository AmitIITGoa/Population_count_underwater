#!/usr/bin/env python3
"""
evaluate_metrics_fixed.py
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# --- YOUR PATHS ---
MODEL_PATH = "/home/amit/experiment_comparing_model/ADVENCE_DETECTION/runs/yolo11l_fish_detection/weights/best.pt"
DATA_YAML  = "/home/amit/experiment_comparing_model/ADVENCE_DETECTION/custom_dataset/data.yaml"

def main():
    parser = argparse.ArgumentParser()
    # We set default=MODEL_PATH so you don't have to type it in CLI
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .pt file")
    parser.add_argument("--data", type=str, default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split: test or val")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Verify paths exist
    if not Path(args.model).exists():
        print(f"[Error] Model not found at: {args.model}")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"[Error] YAML not found at: {args.data}")
        sys.exit(1)

    print(f"Loading model: {Path(args.model).name}...")
    model = YOLO(args.model)

    print(f"Evaluating on '{args.split}' set...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
        save=False
    )

    print("\n" + "="*50)
    print(" 📊 METRICS REPORT")
    print("="*50)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 33)
    print(f"{'Precision':<20} | {metrics.box.mp:.4f}")
    print(f"{'Recall':<20} | {metrics.box.mr:.4f}")
    print(f"{'mAP @ 0.50':<20} | {metrics.box.map50:.4f}")
    print(f"{'mAP @ 0.50:0.95':<20} | {metrics.box.map:.4f}")
    print("="*50)

    # Per-class AP
    if len(metrics.names) > 0:
        print("\n📦 PER-CLASS AP (50-95)")
        for i, name in metrics.names.items():
            # metrics.box.maps is an array of AP50-95 for each class
            ap = metrics.box.maps[i]
            print(f"{name:<20} | {ap:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()