from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")

# Predict on image or folder
model.predict(
    source="All_plans_1024",  # or a new image folder ....
    conf=0.25,                # confidence threshold
    save=True,
    imgsz=1024,
    device="cpu"          
)