from ultralytics import YOLO
import cv2
import numpy as np


def detect_and_save_video(
        input_video_path,
        output_video_path,
        model_name='yolov8n.pt',
        conf_threshold=0.5,
        show_fps=True
):
    """
    Детекция людей на видео с отрисовкой bbox и сохранением результата

    Args:
        input_video_path: путь к входному видео
        output_video_path: путь для сохранения результата
        model_name: модель YOLO (yolov8n.pt, yolov8s.pt, yolov11n.pt и т.д.)
        conf_threshold: порог уверенности (0-1)
        show_fps: показывать ли FPS на видео
    """

    # Загрузка модели
    print(f"Загрузка модели {model_name}...")
    model = YOLO(model_name)

    # Открытие видео
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {input_video_path}")
        return

    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")

    # Создание VideoWriter для сохранения
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID', 'H264'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Счетчики для статистики
    frame_count = 0
    total_detections = 0
    processing_times = []

    print("Начало обработки видео...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Детекция (засекаем время)
        import time
        start_time = time.time()

        results = model(
            frame,
            classes=[0],  # 0 = person
            conf=conf_threshold,
            verbose=False  # отключить вывод в консоль
        )

        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        current_fps = 1 / processing_time if processing_time > 0 else 0

        # Получение детекций
        boxes = results[0].boxes
        total_detections += len(boxes)

        # Отрисовка bbox и подписей
        for box in boxes:
            # Координаты bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Уверенность
            confidence = float(box.conf[0])

            # Класс (должен быть 0 - person)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Цвет bbox (BGR)
            color = (0, 255, 0)  # зеленый

            # Рисуем прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Подготовка текста
            label = f"{class_name}: {confidence:.2f}"

            # Размер текста для фона
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Рисуем фон для текста
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # заполненный прямоугольник
            )

            # Рисуем текст
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # черный текст
                2
            )

        # Добавление информации на кадр
        info_y = 30

        # Количество обнаруженных людей
        people_text = f"People: {len(boxes)}"
        cv2.putText(
            frame,
            people_text,
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # FPS
        if show_fps:
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, info_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Прогресс
        progress = (frame_count / total_frames) * 100
        progress_text = f"Progress: {progress:.1f}%"
        cv2.putText(
            frame,
            progress_text,
            (10, info_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Записываем кадр в выходное видео
        out.write(frame)

        # Прогресс в консоль
        if frame_count % 30 == 0:
            print(f"Обработано кадров: {frame_count}/{total_frames} ({progress:.1f}%)")

    # Освобождение ресурсов
    cap.release()
    out.release()

    # Итоговая статистика
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
    avg_people_per_frame = total_detections / frame_count if frame_count > 0 else 0

    print("\n" + "=" * 50)
    print("СТАТИСТИКА ОБРАБОТКИ")
    print("=" * 50)
    print(f"Всего кадров обработано: {frame_count}")
    print(f"Всего детекций: {total_detections}")
    print(f"Среднее количество людей на кадр: {avg_people_per_frame:.2f}")
    print(f"Среднее время обработки кадра: {avg_processing_time * 1000:.2f} мс")
    print(f"Средний FPS: {avg_fps:.2f}")
    print(f"Видео сохранено: {output_video_path}")
    print("=" * 50)


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Базовый пример
    detect_and_save_video(
        input_video_path='crowd.mp4',
        output_video_path='output_video.mp4',
        model_name='yolov8n.pt',  # или 'yolov11n.pt'
        conf_threshold=0.5,
        show_fps=True
    )

    # Пример сравнения моделей
    """
    models_to_test = ['yolov8n.pt', 'yolov8s.pt', 'yolov11n.pt']

    for model_name in models_to_test:
        output_name = f"output_{model_name.replace('.pt', '')}.mp4"
        print(f"\nТестирование {model_name}...")
        detect_and_save_video(
            input_video_path='input_video.mp4',
            output_video_path=output_name,
            model_name=model_name,
            conf_threshold=0.5
        )
    """