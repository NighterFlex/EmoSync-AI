import os
import kagglehub

# 1. Download/locate the FER-2013 dataset in the global cache
dataset_path = kagglehub.dataset_download("msambare/fer2013")
print("Dataset located at:", dataset_path)

# 2. Use os.path.join to reference the specific folders you need
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")