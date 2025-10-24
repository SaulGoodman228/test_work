import cv2
import numpy as np
import os
import pandas as pd

#Параметры нарезки
video_path = "videos/origin/crowd.mp4"
output_dir = "videos/frames"
num_frames = 100

# Нарезчик видео
def video_cutter(video_path:str,output_dir:str,num_frames:int) -> pd.DataFrame:
    # Создаем папку для кадров
    os.makedirs(output_dir, exist_ok=True)

    #Открываем видео
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Всего кадров: {total_frames}, FPS: {fps}, Длительность: {duration:.2f} сек")

    #Случайные кадры
    np.random.seed(1337)  # фиксируем для повторяемости
    random_frame_ids = np.random.choice(total_frames, num_frames, replace=False)
    random_frame_ids.sort()

    #Сохраняем кадры
    records = []

    for i, frame_id in enumerate(random_frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = cap.read()
        if not success:
            continue

        filename = f"frame_{i:03d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

        time_sec = frame_id / fps
        records.append({"filename": filename, "frame_id": frame_id, "time_sec": round(time_sec, 2)})

    cap.release()

    #Сохраняем CSV с метаданными
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "frames_info.csv"), index=False)

    #Возвращает DF вида имя_фрейма/номер_фрейма/время_появления
    return df

#Функция открытия видео
def video_open(video_path:str) -> cv2.VideoCapture:
    # Открываем видео
    cap = cv2.VideoCapture(video_path)

    #Проверка есть ли файл
    if not cap.isOpened():
        raise FileNotFoundError("Не удалось открыть видеофайл")

    return cap

#Отрисовка определенных объектов на видео с достоверностью.
def draw_detections(frame, boxes, model, extra_info=None):
    """
    Отрисовывает bounding boxes и подписи на кадре.

    Args:
        frame: кадр (numpy.ndarray)
        boxes: detections (results[0].boxes)
        model: YOLO модель (для доступа к именам классов)
        extra_info: dict с дополнительной информацией (например, people_count, progress)
    """
    for box in boxes:
        # Координаты bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Уверенность и класс
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Цвет bbox (BGR)
        color = (0, 255, 0)

        # Рисуем bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Подпись
        label = f"{class_name}: {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Добавляем дополнительную информацию (если есть)
    if extra_info:
        info_y = 30
        cv2.putText(frame, f"People: {extra_info.get('people_count', 0)}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Progress: {extra_info.get('progress', 0):.1f}%", (10, info_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

if __name__ == "__main__":
    data = video_cutter(video_path,output_dir,num_frames)
    data.to_csv('video_info.csv', index=False)