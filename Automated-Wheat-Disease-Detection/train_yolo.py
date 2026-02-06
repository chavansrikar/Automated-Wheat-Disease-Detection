from ultralytics import YOLO

# --- Configuration (Using Absolute Paths) ---
# Path to your .yaml file (which is in your 'r' folder)
DATA_CONFIG_FILE = r'C:\Users\chavan srikar\OneDrive\Desktop\r\wheat.yaml'
# --- End Configuration ---


# Load a pre-trained model
model = YOLO('yolov8n.pt') 

# --- Train the model ---
print(f"Starting YOLOv8 training using {DATA_CONFIG_FILE}")
print("NOTE: You are training on a CPU. This will be VERY slow. Please be patient.")

results = model.train(
    data=DATA_CONFIG_FILE,
    epochs=25,
    imgsz=640,
    batch=8,
    name='wheat_detect_run1_cpu', # New name for this run
    device='cpu'  # We are forcing it to use CPU since you have an AMD GPU
)

print("Training complete!")
print("Your trained model is saved in the 'runs/detect/wheat_detect_run1_cpu' folder.")