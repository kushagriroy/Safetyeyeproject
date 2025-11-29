SafetyEye 👷‍♂️👁️

SafetyEye is an AI-powered computer vision system designed to monitor safety compliance in industrial environments. Utilizing the YOLOv8 object detection architecture, it identifies workers and checks for proper Personal Protective Equipment (PPE) usage in real-time images and video feeds.

The system detects essential gear like hardhats, safety vests, and masks, while simultaneously flagging violations (e.g., "NO-Hardhat", "NO-Safety Vest"). It includes an intelligent Rule Engine with object tracking to log specific violation events per worker to a CSV file.

🚀 Key Features

Real-time Detection: Detects 10 distinct classes including PPE, workers, machinery, and safety cones.

Violation Logic: Custom rule engine specifically flags non-compliance (e.g., detecting a person without a vest).

Object Tracking: Uses unique IDs for workers to track violations across video frames, preventing duplicate alerts for the same event.

Automated Logging: Records all safety breaches with timestamps, worker IDs, and confidence scores to violation_logs.csv.

Data Pipeline: Includes scripts for cleaning datasets, validating labels, and augmenting training data.

🧠 The Model

Architecture: Ultralytics YOLOv8 (Nano)

Training: Trained for 50+ epochs on a custom PPE dataset.

Input Resolution: 640x640

Classes:
0.  Hardhat

Mask

NO-Hardhat ⚠️ (Violation)

NO-Mask ⚠️ (Violation)

NO-Safety Vest ⚠️ (Violation)

Person

Safety Cone

Safety Vest

Machinery

Vehicle

🛠️ Installation & Requirements

This project is optimized for Google Colab but can run locally with a GPU.

Dependencies:

Python 3.8+

ultralytics

torch

opencv-python

pandas

pyyaml

To install dependencies:

pip install ultralytics opencv-python pandas


📂 Project Structure

SAFETYEYE.ipynb: The main Jupyter Notebook containing training, validation, and inference pipelines.

safetyeye_config.yaml: Configuration file defining dataset paths and class names.

violation_logs.csv: Auto-generated log file storing detection events.

runs/: Directory where YOLO saves training weights (best.pt) and inference visualizations.

🖥️ Usage Guide

1. Dataset Setup

The notebook assumes a dataset structure (standard YOLO format) is available. It includes a data cleaning step to remove corrupt image/label pairs before training.

2. Training

To train the model on your own dataset:

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pre-trained weights
results = model.train(
    data='safetyeye_config.yaml',
    epochs=50,
    imgsz=640,
    project='SafetyEye_Project_Run',
    name='initial_run'
)


3. Running Inference with Tracking & Logging

The core functionality lies in the inference loop which applies the Rule Engine.

# Example Logic Snippet
def evaluate_violation(box, class_names):
    label = class_names[int(box.cls)]
    if label == 'NO-Hardhat':
        return "NoHelmet"
    # ... (other rules)
    return None

# Run prediction on video
results = model.track(source="site_video.mp4", stream=True, persist=True)


4. Output

The system generates a visual video file with bounding boxes and a violation_logs.csv file structured as follows:

Timestamp

Violation Type

Worker ID

Confidence

Source

2023-10-25 14:30:01

NoVest

W_12

0.88

text_video.mp4

2023-10-25 14:30:05

NoHelmet

W_4

0.92

text_video.mp4

📈 Performance

The model utilizes data augmentation (HSV adjustments, mosaic, flipping) to improve robustness against varying lighting conditions and angles common in construction sites.

⚠️ Note on Google Colab

If running in Google Colab, ensure you mount your Google Drive to persist the trained model weights (best.pt) and the generated logs, as the runtime storage is temporary.

from google.colab import drive
drive.mount('/content/drive')
