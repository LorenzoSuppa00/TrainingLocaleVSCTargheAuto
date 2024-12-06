from ultralytics import YOLO

# Load the YOLO model (you may use other YOLO versions)
model = YOLO("yolov8n.pt")  # A lightweight YOLO model for faster inference

# Train the model with your dataset
model.train(data="", epochs=10, imgsz=640, batch=16)

# The best weights will be saved as 'best.pt'. Optionally, save them as 'best.bt' as well.
model.save("bestv7.pt")  # explicitly saving as '.bt'


