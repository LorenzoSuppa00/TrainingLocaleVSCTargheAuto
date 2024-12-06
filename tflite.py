from ultralytics import YOLO

# ! Runnare prima da riga 4 a riga 8, Commentare da riga 12 a 15, 
# ! una volta runnato, Commentare da riga 4 a riga 8 e Runnare da riga 12 a riga 15  
# # Load Backup Training model
# model = YOLO("bestv9.pt")

# # Export the model to TFLite format
# model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("bestv9_saved_model/bestv9_float32.tflite")

# Run inference
results = tflite_model("car.jpg")