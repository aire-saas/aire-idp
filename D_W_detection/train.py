from ultralytics import YOLO


model = YOLO("yolov8s.pt")

# Train
model.train(
    data="data.yaml",
    epochs=30,        
    imgsz=1024,
    batch=2,
    workers=0,
    device="cpu"         
)