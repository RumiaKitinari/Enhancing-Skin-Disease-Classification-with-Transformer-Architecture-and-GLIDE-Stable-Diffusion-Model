import kagglehub
import os
import shutil

# 1.) Download HAM10000 dataset
path = kagglehub.dataset_download("shubhamgoel27/dermnet")

print(f"Dataset downloaded to temporary location: {path}")

# 2.) Set directory_destination
target_folder = os.path.join("/data/team01/ds340w/datasets", 
                             "dermnet_data")

# 3.) Move the files from the cache to your project folder
if not os.path.exists(target_folder):
    shutil.move(path, target_folder)
    print(f"Dataset moved to: {target_folder}")
else:
    print(f"Target folder {target_folder} already exists. Skipping move.")

# List files to verify
print("Files in dataset folder:", os.listdir(target_folder))