import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# 1.) Paths
data_dir1 = "/data/team01/ds340w/datasets/dermnet_data_merged"
output_dir = "/data/team01/ds340w/datasets/dermet_split"

data = []
for class_name in os.listdir(data_dir1):
    class_path = os.path.join(data_dir1, class_name)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            # Ensure we only grab images
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    'image_path': os.path.join(class_path, img_name),
                    'dx': class_name,
                    'filename': img_name
                })

df = pd.DataFrame(data)

# 2.) Create the Split (70% Train, 10% Validation, 20% Test)
# Split = 70% Train / 30% (Test + Validation)
train_df, test_val_df = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['dx'], 
    random_state=17
)

# Split = 20% (Test) / 10% (Validation)
val_df, test_df = train_test_split(
    test_val_df, 
    test_size=20/30, 
    stratify=test_val_df['dx'], 
    random_state=17
)

def organize_files(dataframe, split_name):
    for _, row in dataframe.iterrows():
        target_dir = os.path.join(output_dir, split_name, row['dx'])
        os.makedirs(target_dir, exist_ok=True)
        
        destination_path = os.path.join(target_dir, row['filename'])
        shutil.copy2(row['image_path'], destination_path)

print("Organizing Train set...")
organize_files(train_df, "train")
print("Organizing Validation set...")
organize_files(val_df, "val")
print("Organizing Test set...")
organize_files(test_df, "test")

print(f"Done! Your split data is in: {output_dir}")