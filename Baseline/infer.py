from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/yolo11n_exp/weights/best.pt")

# Run validation using data.yaml
metrics = model.val(
    data="data.yaml",
    imgsz=640
)

# Print metrics
print("\n====== Evaluation Metrics ======")
print(f"Precision:      {metrics.box.mp:.4f}")
print(f"Recall:         {metrics.box.mr:.4f}")
print(f"mAP@50:         {metrics.box.map50:.4f}")
print(f"mAP@50-95:      {metrics.box.map:.4f}")
print("================================")
