import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# --- Configuration ---
# 1. Define the paths to your two trained models
YOLO_MODEL_PATH = 'best.pt'
VGG_MODEL_PATH = 'wheat_classifier_vgg19.h5'

# 2. Define the path to your test image
#    You can get an image from your 'global-wheat-detection/dataset/images/val' folder
TEST_IMAGE_PATH = r"C:\Users\chavan srikar\OneDrive\Pictures\fcddfe5064b392e9a1cbe9a3244ff627.jpg"
# 3. Define your class names (MUST match the VGG19 training order)
#    This list now matches your model's output
CLASS_NAMES = [
    'Crown and Root Rot',  # Index 0
    'Healthy Wheat',       # Index 1
    'Leaf Rust',           # Index 2
    'Wheat Loose Smut'     # Index 3
]

# 4. Set the input size for the VGG19 model
IMG_SIZE = (224, 224)
# --- End Configuration ---


# --- 1. Load Both Models ---
print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
# We must run this on CPU
yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')

print(f"Loading VGG19 Classifier model from: {VGG_MODEL_PATH}")
classifier_model = load_model(VGG_MODEL_PATH)
print("Models loaded successfully.")


# --- 2. Load and Prepare the Test Image ---
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"ERROR: Test image not found at: {TEST_IMAGE_PATH}")
    exit()

print(f"Loading test image: {TEST_IMAGE_PATH}")
# We use cv2 to read the image so we can draw on it later
frame = cv2.imread(TEST_IMAGE_PATH)


# --- 3. Run YOLO Detection (Part 1) ---
# 'conf=0.25' means only detect heads with > 25% confidence
print("Running detection...")
results = yolo_model(frame, device='cpu', conf=0.25)

# Loop over each detection
for r in results:
    boxes = r.boxes
    print(f"Found {len(boxes)} wheat heads.")
    
    for box in boxes:
        # Get bounding box coordinates [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # --- 4. Crop the Detected Head ---
        try:
            # Crop the head from the original frame
            cropped_head = frame[y1:y2, x1:x2]
            
            # --- 5. Pre-process the Crop for VGG19 (Part 2) ---
            # Resize the crop to the required 224x224
            img_resized = cv2.resize(cropped_head, IMG_SIZE)
            
            # Convert to an array and add a "batch" dimension
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalize the pixel values (just like in training)
            img_array /= 255.0
            
            # --- 6. Run VGG19 Classification (Part 2) ---
            prediction = classifier_model.predict(img_array)
            
            # Get the class with the highest probability
            predicted_class_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100
            
            # --- 7. Draw the Results on the Image ---
            label = f'{predicted_class}: {confidence:.1f}%'
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the label text
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        except Exception as e:
            print(f"Error processing box: {e}")

# --- 8. Save and Show the Final Image ---
OUTPUT_IMAGE_PATH = 'pipeline_result.jpg'
cv2.imwrite(OUTPUT_IMAGE_PATH, frame)

print(f"\n--- Process Complete! ---")
print(f"Final image saved to: {OUTPUT_IMAGE_PATH}")
print("You can open 'pipeline_result.jpg' to see the results.")