from ultralytics import YOLO
import cv2

def read_and_preprocess_image(image_path):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (640, 640))  # Resize to the input size of MobileNetV2
    return image


def yolo_train(cv_model:str, data,epochs:int):
    # Загрузка нужной модели yolov8n yolov11n
    model = YOLO(cv_model)
    # Display model information (optional)
    model.info()
    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=data, epochs=epochs, imgsz=640)


yolo_train('yolov8n.pt', )