from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="dQ1UtEnqKt7OEDr73fVS")
project = rf.workspace("prima-prova-roboflow").project("license-plates-detection-8g9l8")
version = project.version(9)
dataset = version.download("yolov11")

modello = YOLO("yolo11n.pt")

modello.train(data="data.yaml", epochs = 10, imgsz = 640, batch = 16)

modello.save("bestv9.pt")



