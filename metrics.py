from ultralytics import YOLO

model = YOLO("yolov8s.pt")
metrics = model.val(data="videos/test_dataset/data.yaml", split="test")

print(metrics)