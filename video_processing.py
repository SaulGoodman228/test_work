
from ultralytics import YOLO
import cv2
import numpy as np
import time
from utils import video_open, draw_detections

def detect_and_save_video(input_video_path: str,
                          output_video_path: str,
                          model_name: str,
                          conf_threshold: float,
                          scale_factor: float = 1.0) -> None:
    """
    Детекция людей на видео с отрисовкой bbox и сохранением результата

    input_video_path - путь к исходному видео
    output_video_path - путь для сохранения видео
    model_name - имя используемой модели
    conf_threshold - порог достоверности
    scale_factor - масштаб видео (0.5 = в 2 раза меньше)
    """
    print(f"Загрузка модели {model_name}...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    processing_times = []

    print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
    print("Начало обработки...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Масштабирование кадра
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (width, height))

        frame_count += 1
        start_time = time.time()

        results = model(frame, classes=[0], conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        total_detections += len(boxes)

        progress = (frame_count / total_frames) * 100
        frame = draw_detections(frame, boxes, model, {
            "people_count": len(boxes),
            "progress": progress})

        out.write(frame)

        processing_times.append(time.time() - start_time)

        if frame_count % 30 == 0:
            print(f"Обработано {frame_count}/{total_frames} ({progress:.1f}%)")

    cap.release()
    out.release()

    avg_time = np.mean(processing_times)
    print(f"\nСредний FPS: {1 / avg_time:.2f}, среднее число людей на кадр: {total_detections / frame_count:.2f}")
    print(f"Видео сохранено: {output_video_path}")


# ИСПОЛЬЗОВАНИЕ
if __name__ == "__main__":
    detect_and_save_video(
        input_video_path='videos/origin/crowd.mp4',
        output_video_path="result.mp4",
        model_name='yolov8s.pt',
        conf_threshold=0.3,
        scale_factor=0.4
    )