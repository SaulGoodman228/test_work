import kagglehub

# Download latest version
path = kagglehub.dataset_download("luiscrmartins/surveillance-images-for-person-detection")

print("Path to dataset files:", path)

DATASET_PATH = path