import os
import shutil
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

print("Starting data split...")

# --- Configuration (Using Absolute Paths) ---
# Path to the folder containing all your training images
IMAGE_SOURCE_DIR = r"C:\Users\chavan srikar\Downloads\global-wheat-detection\train" 

# Path to the folder containing all your .txt label files (created in Step 2)
LABEL_SOURCE_DIR = r"C:\Users\chavan srikar\Downloads\global-wheat-detection\labels" 

# Path where the NEW, organized 'dataset' folder will be created
BASE_OUTPUT_DIR = r"C:\Users\chavan srikar\Downloads\global-wheat-detection\dataset"

VAL_SPLIT_SIZE = 0.2
# --- End Configuration ---

# 1. Create directories
print("Creating directories...")
dir_paths = {
    "images_train": os.path.join(BASE_OUTPUT_DIR, "images", "train"),
    "images_val": os.path.join(BASE_OUTPUT_DIR, "images", "val"),
    "labels_train": os.path.join(BASE_OUTPUT_DIR, "labels", "train"),
    "labels_val": os.path.join(BASE_OUTPUT_DIR, "labels", "val")
}
for path in dir_paths.values():
    os.makedirs(path, exist_ok=True)

# 2. Get all image paths
all_image_files = glob.glob(os.path.join(IMAGE_SOURCE_DIR, "*.jpg"))
print(f"Found {len(all_image_files)} total images.")

# 3. Split the files
train_files, val_files = train_test_split(all_image_files, test_size=VAL_SPLIT_SIZE, random_state=42)
print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

# 4. Helper function to copy files
def copy_files(file_list, image_dest_dir, label_dest_dir):
    missing_labels = 0
    for img_path in tqdm(file_list):
        base_filename = os.path.basename(img_path)
        name_part = os.path.splitext(base_filename)[0]
        label_filename = f"{name_part}.txt"
        label_path = os.path.join(LABEL_SOURCE_DIR, label_filename)
        
        new_img_path = os.path.join(image_dest_dir, base_filename)
        new_label_path = os.path.join(label_dest_dir, label_filename)
        
        shutil.copyfile(img_path, new_img_path)
        
        if os.path.exists(label_path):
            shutil.copyfile(label_path, new_label_path)
        else:
            # This is normal for images that have no wheat heads
            missing_labels += 1
            # We must create an empty .txt file for YOLO
            open(new_label_path, 'w').close()
            
    if missing_labels > 0:
        print(f"Note: {missing_labels} images had no labels (this is normal for background images).")

# 5. Copy files
print("Copying training files...")
copy_files(train_files, dir_paths["images_train"], dir_paths["labels_train"])
print("Copying validation files...")
copy_files(val_files, dir_paths["images_val"], dir_paths["labels_val"])

print("Data splitting complete!")
print(f"Your new dataset is ready in: {BASE_OUTPUT_DIR}")