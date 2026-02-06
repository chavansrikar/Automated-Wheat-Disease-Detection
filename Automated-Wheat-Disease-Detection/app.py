import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- 1. Load Your Models (Load them ONCE, globally) ---
print("Loading models... This might take a moment.")
YOLO_MODEL_PATH = 'best.pt'
VGG_MODEL_PATH = 'wheat_classifier_vgg19.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Crown and Root Rot',  # Index 0
    'Healthy Wheat',       # Index 1
    'Leaf Rust',           # Index 2
    'Wheat Loose Smut'     # Index 3
]

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
classifier_model = load_model(VGG_MODEL_PATH)
print("Models loaded successfully.")

# --- 2. Define Your Prediction Function ---
# This function takes one input (the image) and returns one output (the processed image)
def detect_and_classify(input_image):
    if input_image is None:
        return None

    # Gradio provides the image as RGB, but OpenCV needs BGR
    frame = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # --- 3. Run Your Pipeline (Copied from run_pipeline.py) ---
    print("Running detection...")
    results = yolo_model(frame, device='cpu', conf=0.25)
    
    # Loop over each detection
    for r in results:
        boxes = r.boxes
        print(f"Found {len(boxes)} wheat heads.")
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            try:
                # Crop, Resize, Normalize
                cropped_head = frame[y1:y2, x1:x2]
                img_resized = cv2.resize(cropped_head, IMG_SIZE)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                # Predict
                prediction = classifier_model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100
                
                # Draw results on the image
                label = f'{predicted_class}: {confidence:.1f}%'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing box: {e}")

    # --- 4. Return the Final Image ---
    # Convert the final image from BGR back to RGB for Gradio to display
    print("Processing complete.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# --- 5. Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="numpy", label="Upload Wheat Field Image"),
    outputs=gr.Image(type="numpy", label="Analysis Result"),
    title="Automated Wheat Disease Detection",
    description="This app uses a YOLO model to find wheat heads and a VGG19 model to classify their disease. (Project by Chavan Srikar)",
    examples=[
        # Add paths to any example images you have
        # e.g., 'global-wheat-detection/dataset/images/val/0021b70a9.jpg'
    ]
)

if __name__ == "__main__":
    iface.launch()