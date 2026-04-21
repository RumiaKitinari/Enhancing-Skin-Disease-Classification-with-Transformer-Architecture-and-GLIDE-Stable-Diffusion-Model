# Merge datasets with "train", "test", and/or "validation" splits

import os
import shutil

train_dir = '/data/team01/ds340w/datasets/Atlas_ISIC_data/Atlas dan ISIC2019 (31 classes)/train' 
test_dir = '/data/team01/ds340w/datasets/Atlas_ISIC_data/Atlas dan ISIC2019 (31 classes)/test' 
merged_dir = '/data/team01/ds340w/datasets/Atlas_ISIC_merged'

def merge_datasets(source_dirs, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
        print(f"Created destination directory: {destination}")

    for source in source_dirs:
        # Get list of all subfolders (classes)
        classes = os.listdir(source)
        
        for class_name in classes:
            source_class_path = os.path.join(source, class_name)
            dest_class_path = os.path.join(destination, class_name)

            # Create the class subfolder in the merged directory if it doesn't exist
            if not os.path.exists(dest_class_path):
                os.makedirs(dest_class_path)

            # Copy all files from the source class folder to the merged class folder
            files = os.listdir(source_class_path)
            for file_name in files:
                src_file = os.path.join(source_class_path, file_name)
                dst_file = os.path.join(dest_class_path, file_name)
                
                # Use copy2 to preserve metadata (like timestamps)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Skipping directory: {src_file}")
        
        print(f"Finished processing: {source}")

# Execute the merge
merge_datasets([train_dir, test_dir], merged_dir)
print(f"All images successfully merged into {merged_dir}")