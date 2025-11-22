

# 🌸 Flower Blooming Detection using YOLOv8

## 📌 Project Overview

This project implements a **real-time flower blooming detection system** using **YOLOv8 (You Only Look Once v8)**, a state-of-the-art object detection model.  
The system detects and classifies flowers into two categories: **“Bloomed”** and **“Unbloomed”**, developed during a focused **10-day industrial skill program** to gain hands-on experience with modern computer vision pipelines and cloud-based data management tools.

***

## 📚 Table of Contents

- [Dataset Creation and Annotation](#1-dataset-creation-and-annotation)  
- [Project Structure](#2-project-structure)  
- [Installation](#3-installation)  
- [Training Process](#4-training-process)  
- [Testing and Inference](#5-testing-and-inference)   
- [Results](#6-results)  
- [Future Improvements](#7-future-improvements)  
- [How to Run (Deployment Export)](#8-how-to-run-deployment-export)   
- [Acknowledgments](#acknowledgments)  
- [License](#license)  

***

## 1. Dataset Creation and Annotation

### Data Source & License

- **Platform:** Roboflow  
  https://universe.roboflow.com/flower-blooming-detection/flower-blooming-yolov9  
- **Dataset Version:** v1 (2025-05-20)  
- **License:** **CC BY 4.0**  

### Data Details

- **Total Images:** 508 source images  
- **Target Classes:**  
  - `Bloomed`  
  - `Unbloomed`  
- **Dataset Split:**  
  - Train: 70%  
  - Validation: 20%  
  - Test: 10%  

### Preprocessing and Augmentation

| Step          | Technique           | Detail                                              |
|--------------|---------------------|-----------------------------------------------------|
| Preprocessing| Auto-orientation    | Stripped EXIF data                                  |
| Preprocessing| Resize              | All images resized to **640×640** (stretch)        |
| Augmentation | Salt and Pepper     | Applied to 0.1% of pixels, 3 augmented versions     |

### Dataset Structure

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

***

## 2. Project Structure

```text
flower-blooming-detection/
├── data.yaml              # YOLO dataset configuration
├── train.py               # Script for model training
├── test.py                # Script for model inference
├── requirements.txt       # Python dependencies
└── runs/                  # Training results
    └── detect/
        └── train/
            └── weights/
                ├── best.pt  # Best model weights
                └── last.pt  # Last epoch weights
```

### `data.yaml` Content

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['Bloomed', 'Unbloomed']

roboflow:
  workspace: flower-blooming-detection
  project: flower-blooming-yolov9
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/flower-blooming-detection/flower-blooming-yolov9/dataset/1
```

***

## 3. Installation

### Clone the Repository

```bash
git clone [repository-url]
cd flower-blooming-detection
```

### (Recommended) Create a Virtual Environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python numpy Pillow PyYAML tqdm matplotlib seaborn pandas scikit-learn
```

Or use:

```bash
pip install -r requirements.txt
```

***

## 4. Training Process

### Model Selection

- Base model: **YOLOv8n (nano)** – optimized for **speed and efficiency**  
- Pretrained on **COCO** for transfer learning  

### Training Configuration

- **Epochs:** 20  
- **Batch size:** 8 (CPU-optimized)  
- **Image size:** 640×640  
- **Device:** CPU  
- **Confidence threshold:** 0.25  

### Training Code Example

```python
from ultralytics import YOLO

# Load a base model (example: YOLOv8c)
model = YOLO('yolov8c.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=20,
    imgsz=640,
    batch=8,
    patience=10,
    save=True,
    device='cpu',
    workers=2,
    cache=True
)
```

***

## 5. Testing and Inference

### Run Test Script

```bash
# Test on default image/video configured in test.py
python test.py

# Test on a specific video file
python test.py --source path/to/video.mp4
```

### Inference Code Example

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to test image
test_image_path = 'test/images/example_flower.jpg'

# Run inference
results = model.predict(test_image_path, conf=0.25)

# (Add your code here to visualize and save results)
```

### Model Performance

- **Classes detected:** `Bloomed`, `Unbloomed`  

***

## 6. Results

Check out the flower blooming detection result here:  
[Flower Blooming Detection Demo Video](https://drive.google.com/file/d/1oLfPQuPtbBZ9-EZvjfdCDJPELskPkexN/view?usp=drive_link)  

*(Click the link to view the result on Google Drive)*

***

## 7. Future Improvements

- Train for more epochs on a **GPU** for higher accuracy.  
- Implement **real-time video stream detection** for live monitoring.  
- Integrate into an **automated flower monitoring system** (e.g., greenhouse or garden IoT).  

***

## 8. How to Run (Deployment Export)

YOLOv8 supports export to multiple formats for deployment (ONNX, TensorRT, etc.).

### Export to ONNX

```bash
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.export(format='onnx')"
```

***

## Acknowledgments

- **YOLOv8** by Ultralytics  
- **Roboflow** for dataset management and versioning  
- **College / Program Name** and **Industry Mentors** for guidance and support  

***

## License

- **Dataset:** **CC BY 4.0**, as per Roboflow dataset license.  

