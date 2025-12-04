#!/usr/bin/env python3
"""
train_yolo11l_enhanced.py

YOLOv11l Fish Detection Training Pipeline (Enhanced)
----------------------------------------------------
✔ Full YOLOv11l training, validation, and test evaluation
✔ JSON + image predictions saved neatly
✔ Hooks for advanced future enhancements:
   - ONNX / TensorRT export
   - Quantization (Dynamic)
   - Semi-supervised pseudo-labeling
   - Depth fusion
   - SAM annotation assistance
   - Continual learning (EWC stub)
   - Temporal tracking (ByteTrack stub)
   - Grad-CAM / Explainable AI (stub)
   - Flask API deployment (stub)
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Optional
import torch
from ultralytics import YOLO

# ==========================================================
# CONFIG
# ==========================================================
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "custom_dataset"
RUNS_DIR = ROOT / "runs"
PRED_DIR = ROOT / "predictions"
JSON_RESULTS_PATH = PRED_DIR / "yolo11l_test_results.json"

MODEL_SOURCE = "yolo11l.pt"
EPOCHS = 100
BATCH = 8
IMGSZ = 640
SEED = 42
WORKERS = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def log(msg): print(f"[+]: {msg}")
def warn(msg): print(f"[!]: {msg}", file=sys.stderr)

# ==========================================================
# 1️⃣ VERIFY DATASET STRUCTURE
# ==========================================================
def verify_structure(data_dir: Path):
    log("Verifying dataset structure...")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]:
        p = data_dir / sub
        if not p.exists():
            warn(f"Missing: {p}")
        else:
            log(f"{sub}: {len(list(p.glob('*.*')))} files")
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists(): log(f"data.yaml found at {yaml_path}")
    else: warn("data.yaml not found!")
    log("Structure verification complete.\n")

# ==========================================================
# 2️⃣ VERIFY/CREATE DATA.YAML
# ==========================================================
def verify_data_yaml(data_dir: Path) -> str:
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f: cfg = yaml.safe_load(f)
        log(f"Using existing data.yaml ({yaml_path})")
    else:
        data = {
            "path": str(data_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": ["fish"]
        }
        with open(yaml_path, "w") as f: yaml.dump(data, f)
        log(f"Created new data.yaml at {yaml_path}")
    return str(yaml_path)

# ==========================================================
# 3️⃣ TRAINING
# ==========================================================
def train_yolo11l(data_yaml, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ):
    log("Starting YOLOv11l training...")
    model = YOLO(MODEL_SOURCE)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="yolo11l_fish_detection",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0,
        cos_lr=True, close_mosaic=10,
        patience=50, save=True, val=True, plots=True,
        exist_ok=True, pretrained=True,
        cache=False, workers=WORKERS,
        rect=False, resume=False, amp=True,
    )
    log(f"Training complete. Results: {results.save_dir}")
    return results

# ==========================================================
# 4️⃣ VALIDATION
# ==========================================================
def validate_model(model_path, data_yaml):
    log(f"Validating model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split='val', imgsz=IMGSZ, batch=BATCH, conf=0.25, iou=0.6, device=DEVICE, plots=True)
    try:
        log(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  Prec={metrics.box.mp:.4f}  Rec={metrics.box.mr:.4f}")
    except Exception: log("Validation metrics format changed; printing raw metrics.")
    return metrics

# ==========================================================
# 5️⃣ TEST + JSON RESULTS
# ==========================================================
def test_and_save_json(model_path, test_dir, data_yaml):
    log("Running inference on test set...")
    model = YOLO(model_path)
    if (DATASET_DIR / "test" / "images").exists():
        tm = model.val(data=data_yaml, split='test', imgsz=IMGSZ, batch=BATCH, conf=0.25, iou=0.6, device=DEVICE)
        log(f"Test mAP50={tm.box.map50:.4f}, mAP50-95={tm.box.map:.4f}")

    results = model.predict(
        source=str(test_dir), save=True, save_txt=True, save_conf=True,
        conf=0.25, iou=0.45, imgsz=IMGSZ, device=DEVICE,
        project=str(PRED_DIR), name="yolo11l_test_results", exist_ok=True,
    )

    all_preds = []
    for res in results:
        path = str(getattr(res, "path", "<unknown>"))
        entry = {"image_path": path, "image_name": Path(path).name, "detections": []}
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            for (x1, y1, x2, y2), conf, cls_id in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy()
            ):
                entry["detections"].append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": model.names[int(cls_id)]
                })
        all_preds.append(entry)

    JSON_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_RESULTS_PATH, "w") as f: json.dump(all_preds, f, indent=4)
    log(f"Saved JSON: {JSON_RESULTS_PATH}")
    log(f"Total images: {len(all_preds)} | With detections: {sum(len(p['detections'])>0 for p in all_preds)}")

# ==========================================================
# 6️⃣ EXPORT + ADVANCED METHODS
# ==========================================================
def export_onnx(model_path, out_path):
    log("Exporting to ONNX...")
    try:
        model = YOLO(model_path)
        model.export(format="onnx", imgsz=IMGSZ, opset=12)
        log("ONNX export complete.")
    except Exception as e:
        warn(f"ONNX export failed: {e}")

def quantize_model_dynamic(model_path, out_path):
    log("Applying dynamic quantization...")
    try:
        model = torch.load(model_path, map_location="cpu")
        q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(q_model, str(out_path))
        log(f"Quantized model saved at {out_path}")
    except Exception as e:
        warn(f"Quantization failed: {e}")

def pseudo_label_step(unlabeled_dir, model_w, conf_th=0.6):
    log("Running pseudo-labeling...")
    model = YOLO(model_w)
    res = model.predict(source=str(unlabeled_dir), imgsz=IMGSZ, conf=conf_th, device=DEVICE)
    out_labels = unlabeled_dir.parent / "pseudo_labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    for r in res:
        p = Path(r.path)
        file = out_labels / f"{p.stem}.txt"
        lines = []
        if getattr(r, "boxes", None) is None: continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clsids = r.boxes.cls.cpu().numpy()
        for (x1,y1,x2,y2), conf, cid in zip(xyxy, confs, clsids):
            if conf < conf_th: continue
            lines.append(f"{int(cid)} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}")
        open(file, "w").write("\n".join(lines))
    log(f"Pseudo-labels saved in {out_labels}")

# === Stubs (safe placeholders for future integration) ===
def depth_fusion_stub(*_): warn("Depth fusion not implemented yet.")
def sam_annotation_helper(*_): warn("SAM annotation helper not implemented yet.")
def continual_learning_stub(*_): warn("Continual learning not implemented yet.")
def tracking_stub(*_): warn("Tracking not implemented yet.")
def gradcam_stub(*_): warn("Grad-CAM not implemented yet.")
def api_export_stub(model_path, export_dir):
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "app.py").write_text(f'''
from flask import Flask, request, jsonify
from ultralytics import YOLO
app = Flask(__name__)
model = YOLO("{model_path}")
@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({{"error": "no image"}}), 400
    img = request.files["image"]
    r = model.predict(img, conf=0.25)
    return jsonify({{"detections": len(r[0].boxes) if r else 0}})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
''')
    log(f"API scaffold written to {export_dir}/app.py")

# ==========================================================
# 7️⃣ MAIN PIPELINE
# ==========================================================
def main(argv=None):
    global MODEL_SOURCE  # ✅ FIXED HERE

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", "-e", type=int, default=EPOCHS)
    p.add_argument("--batch", "-b", type=int, default=BATCH)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--model", type=str, default=MODEL_SOURCE)
    p.add_argument("--data", type=str, default=str(DATASET_DIR / "data.yaml"))
    p.add_argument("--export-onnx", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--pseudo", action="store_true")
    p.add_argument("--use-depth", action="store_true")
    p.add_argument("--sam-annot", action="store_true")
    p.add_argument("--continual", action="store_true")
    p.add_argument("--tracking", action="store_true")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--api-export", action="store_true")
    args = p.parse_args(argv)

    MODEL_SOURCE = args.model  # ✅ after global declaration

    log("="*80)
    log("🐟 YOLOv11l Fish Detection Training Pipeline (Enhanced)")
    log(f"Device: {DEVICE}")
    log("="*80)

    verify_structure(DATASET_DIR)
    data_yaml = args.data
    if not Path(data_yaml).exists(): data_yaml = verify_data_yaml(DATASET_DIR)

    results = train_yolo11l(data_yaml=data_yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)

    save_dir = getattr(results, "save_dir", RUNS_DIR / "yolo11l_fish_detection")
    best_model = Path(save_dir) / "weights" / "best.pt"
    if not best_model.exists():
        best_model = Path(save_dir) / "weights" / "last.pt"

    if best_model.exists():
        validate_model(str(best_model), data_yaml)
        test_dir = DATASET_DIR / "test" / "images"
        if test_dir.exists(): test_and_save_json(str(best_model), test_dir, data_yaml)
    else:
        warn("Best model not found!")

    # ---- Optional Enhancements ----
    if args.export_onnx: export_onnx(str(best_model), RUNS_DIR / "exports" / "yolo11l.onnx")
    if args.quantize: quantize_model_dynamic(str(best_model), RUNS_DIR / "exports" / "yolo11l_quant.pt")
    if args.pseudo: pseudo_label_step(DATASET_DIR / "unlabeled", str(best_model))
    if args.use_depth: depth_fusion_stub()
    if args.sam_annot: sam_annotation_helper()
    if args.continual: continual_learning_stub()
    if args.tracking: tracking_stub()
    if args.gradcam: gradcam_stub()
    if args.api_export: api_export_stub(str(best_model), RUNS_DIR / "api_export")

    log("="*80)
    log("✅ TRAINING PIPELINE COMPLETE")
    log("="*80)


if __name__ == "__main__":
    main()
