from ultralytics import YOLO
import torch

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='data.yaml',
    epochs=20,  # reduced epochs for faster training
    imgsz=640,
    batch=8,  # reduced batch size for CPU
    patience=10,
    save=True,
    device='cpu',  # explicitly use CPU
    workers=2,  # reduced workers for CPU
    cache=True  # cache images for faster training
)

# Save the trained model
model.save('flower_detection_model.pt') 