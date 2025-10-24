import kagglehub

# Загрузка датасета с людьми на уличных камерах
path = kagglehub.dataset_download("constantinwerner/human-detection-dataset")
print("Path to dataset files:", path)
