import os
import shutil
import re


# 1.) Paths
source_directory = '/data/team01/ds340w/datasets/Atlas_merged' 
target_base_directory = '/data/team01/ds340w/datasets/Atlas_categorized'

# 2.) Target List
target_classes = [
    "Leprosy Lepromatous", "Darier_s Disease", "Pityriasis Rosea", 
    "pigmented benign keratosis", "Hailey-Hailey Disease", "Melanoma", 
    "Lupus Erythematosus Chronicus Discoides", "vascular lesion", 
    "Tinea Corporis", "Mycoses Fungoides", "Papilomatosis Confluentes And Reticulate", 
    "Lichen Planus", "actinic keratosis", "squamous cell carcinoma", 
    "Neurofibromatosis", "Leprosy Tuberculoid", "Basal Cell Carcinoma", 
    "Porokeratosis Actinic", "Impetigo", "Epidermolysis Bullosa Pruriginosa", 
    "Herpes Simplex", "Tinea Nigra", "Pediculosis Capitis", 
    "Molluscum Contagiosum", "seborrheic keratosis", "dermatofibroma", 
    "Psoriasis", "Tungiasis", "Leprosy Borderline", "nevus", "Larva Migrans"
]

# Create a normalized version for more robust matching (lowercase)
normalized_targets = {c.lower().strip(): c for c in target_classes}

def filter_and_organize():
    # Ensure target directory exists
    if not os.path.exists(target_base_directory):
        os.makedirs(target_base_directory)

    # Process each class
    for class_label in target_classes:
        # Create a directory for this specific class
        class_output_dir = os.path.join(target_base_directory, class_label.replace(" ", "_"))
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Break class into individual words for matching (e.g., ["Basal", "Cell", "Carcinoma"])
        # We use lowercase to make the search case-insensitive
        keywords = class_label.lower().split()
        
        print(f"Searching for images matching: {class_label}...")

        # Walk through source_directory recursively
        for root, dirs, files in os.walk(source_directory):
            for filename in files:
                # Check if it's an image file
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue
                
                # Check if ALL words in the class label appear in the filename
                # Example: 'basal_cell_123.jpg' matches 'Basal Cell Carcinoma' 
                # if we look for any keywords, but here we check for a subset match
                lower_filename = filename.lower()
                
                # Logic: Match if the full class name string exists or all parts exist
                if all(word in lower_filename for word in keywords):
                    src_path = os.path.join(root, filename)
                    dst_path = os.path.join(class_output_dir, filename)
                    
                    # Copying to avoid data loss in source
                    shutil.copy2(src_path, dst_path)

    print(f"\nFiltering complete. Files are organized in: {target_base_directory}")

if __name__ == "__main__":
    filter_and_organize()