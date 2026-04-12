import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# 1.) Paths
data_dir = "/data/team01/ds340w/datasets/ham10000_data"
metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
output_dir = "/data/team01/ds340w/datasets/ham10000_split"

# 2.) Load Metadata
df = pd.read_csv(metadata_path)

# 3.) Create the Split (70% Train, 30% Test)
# train_df, test_df = train_test_split(df, test_size=0.30, stratify=df['dx'], random_state=42)

# 3.) Create the Split (70% Train, 10% Validation, 20% Test)
# Split = 70% Train / 30% (Test + Validation)
train_df, test_val_df = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['dx'], 
    random_state=17
)

# # Split = 20% (Test) / 10% (Validation)
test_df, val_df = train_test_split(
    test_val_df, 
    test_size=20/30, 
    stratify=test_val_df['dx'], 
    random_state=17
)

def organize_files(dataframe, split_name):
    for _, row in dataframe.iterrows():
        image_id = row['image_id']
        label = row['dx']
        
        # Create folder for the class 
        target_path = os.path.join(output_dir, split_name, label)
        os.makedirs(target_path, exist_ok=True)
        
        # Find where the image is (checking part_1 + part_2 folders)
        found = False
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            source_file = os.path.join(data_dir, part, f"{image_id}.jpg")
            if os.path.exists(source_file):
                shutil.copy(source_file, os.path.join(target_path, f"{image_id}.jpg"))
                found = True
                break
        
        if not found:
            print(f"Warning: {image_id} not found in parts 1 or 2.")

# 4.) Status Updates
print("Organizing Train set...")
organize_files(train_df, "train")
print("Organizing Test set...")
organize_files(test_df, "test")
print("Organizing Validation set...")
organize_files(val_df, "val")
print(f"Done! Your split data is in: {output_dir}")