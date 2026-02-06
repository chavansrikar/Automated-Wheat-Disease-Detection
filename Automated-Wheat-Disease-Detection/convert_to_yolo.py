import pandas as pd
import os
from tqdm import tqdm
import json

# --- Configuration (Using Absolute Paths) ---
# Path to the CSV file in your Downloads
CSV_FILE = r"C:\Users\chavan srikar\Downloads\global-wheat-detection\train.csv"

# Path where the NEW 'labels' folder will be created
LABEL_OUTPUT_DIR = r"C:\Users\chavan srikar\Downloads\global-wheat-detection\labels" 

# --- End Configuration ---

# Create the output directory if it doesn't exist
os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

# --- Load the CSV ---
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: Cannot find the file at {CSV_FILE}")
    print("Please check the path and try again.")
    exit()

# The image size is constant
IMG_WIDTH = 1024.0
IMG_HEIGHT = 1024.0

# --- Helper function ---
def convert_to_yolo(bbox):
    box = json.loads(bbox)
    xmin = box[0]
    ymin = box[1]
    w = box[2]
    h = box[3]
    x_center = xmin + (w / 2)
    y_center = ymin + (h / 2)
    
    # Normalize
    x_center_norm = x_center / IMG_WIDTH
    y_center_norm = y_center / IMG_HEIGHT
    w_norm = w / IMG_WIDTH
    h_norm = h / IMG_HEIGHT
    
    class_index = 0
    return f"{class_index} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"

# --- Main Conversion Loop ---
print(f"Reading CSV from: {CSV_FILE}")
print(f"Saving labels to: {LABEL_OUTPUT_DIR}")

image_groups = df.groupby('image_id')
for image_name, group in tqdm(image_groups):
    label_filename = f"{image_name}.txt"
    label_filepath = os.path.join(LABEL_OUTPUT_DIR, label_filename)
    
    with open(label_filepath, 'w') as f:
        for _, row in group.iterrows():
            yolo_line = convert_to_yolo(row['bbox'])
            f.write(yolo_line + '\n')

print(f"Conversion complete! {len(image_groups)} label files saved.")