import cv2
from ultralytics import YOLO
import joblib
import tkinter as tk
from traffic_light_gui import TrafficLightApp  # Assuming this is your GUI import

# Load AI models for traffic density and green light time prediction
density_model = joblib.load("traffic_density_model.pkl")  # Traffic density model
green_light_model = joblib.load("green_light_time_model.pkl")  # Green light time model
model = YOLO("yolov8n.pt")  # YOLO model for vehicle detection

# Load the video
video_path = r"C:\\Users\\lenovo\\Downloads\\WhatsApp Video 2025-04-24 at 21.58.05_b950991d.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# GUI setup
gui_root = tk.Tk()
traffic_gui = TrafficLightApp(gui_root)

# Function to update the GUI
def update_gui():
    gui_root.after(100, update_gui)

update_gui()
gui_root.after(0, traffic_gui.set_light)

# Setup resizable OpenCV window
cv2.namedWindow("AI Traffic Monitor", cv2.WINDOW_NORMAL)

# Confidence threshold for YOLO detection
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for better display
    frame = cv2.resize(frame, (1280, 720))

    # Perform YOLO vehicle detection
    results = model(frame)[0]

    # Initialize vehicle counts
    vehicle_count = 0
    heavy_count = 0
    motorbike_count = 0

    # Iterate over the detected boxes to count vehicles
    for box in results.boxes:
        cls = model.names[int(box.cls)]  # Get the detected class name
        confidence = box.conf  # Get confidence score

        # Skip objects with low confidence
        if confidence < confidence_threshold:
            continue

        if cls in ['car', 'truck', 'bus', 'motorbike']:  # Filter vehicle classes
            vehicle_count += 1
            if cls in ['bus', 'truck']:  # Identify heavy vehicles
                heavy_count += 1
            if cls == 'motorbike':  # Track motorbikes separately
                motorbike_count += 1

            # Draw bounding boxes around detected vehicles
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Predict traffic density (Low, Medium, High) based on vehicle count
    prediction = density_model.predict([[vehicle_count, heavy_count]])[0]
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    density_label = label_map[prediction]
    traffic_gui.set_light(density_label)

    # Predict the green light time based on vehicle count and heavy count
    green_time = green_light_model.predict([[vehicle_count, heavy_count]])[0]

    # Display traffic density, motorbike count, and green light time on the frame
    cv2.putText(frame, f"Traffic Density: {density_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(frame, f"Green Light Time: {green_time} sec", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("AI Traffic Monitor", frame)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
