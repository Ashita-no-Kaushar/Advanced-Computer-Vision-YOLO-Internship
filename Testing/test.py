from ultralytics import YOLO
import cv2
import os

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to test image
test_image_path = 'test/images/6606749757_b98a4ba403_jpg.rf.16d840a4e41d58d9539356280e738a76.jpg'

# Run inference
results = model.predict(test_image_path, conf=0.25)  # confidence threshold 0.25

# Get the first result
result = results[0]

# Create output directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Save the result with bounding boxes
output_path = 'predictions/test_result.jpg'
result.save(output_path)

print(f"Prediction saved to {output_path}")

# Print detections
for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = result.names[class_id]
    print(f"Detected {class_name} with confidence: {confidence:.2f}") 