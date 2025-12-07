from ultralytics import YOLO

# ==========================================
#  LOAD MODEL (YOLO11n)
# ==========================================
model = YOLO("yolo11n.pt")        # download + load YOLO11 nano

# ==========================================
#  TRAIN MODEL
# ==========================================
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="yolo11n_exp",
    save=True,            # saves last.pt and best.pt (default True)
    save_period=-1,       # don't save every epoch
)
