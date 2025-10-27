from ultralytics import YOLO
from dataset import DATASET_PATH

def main():
    # Загрузка модели
    model = YOLO("yolov8s.pt")

    # Базовая тренировка
    results = model.train(
        data=f"{DATASET_PATH}/person-3/data.yaml",  # путь к data.yaml
        epochs=25,                 # количество эпох
        imgsz=640,                 # размер изображения
        batch=16,                  # размер батча
        device=0,              # 'cpu' или 'cuda' если есть GPU
        workers=2,                 # количество воркеров
        save=True,                 # сохранять чекпоинты
        project='runs/detect',     # папка для результатов
        name='yolov8n_train',
    )

    print("Тренировка завершена!")

if __name__ == "__main__":
    main()

