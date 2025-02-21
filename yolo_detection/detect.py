from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (use 'yolov8n.pt' for faster inference, 'yolov8x.pt' for accuracy)
model = YOLO("yolov8n.pt")

# Load video
video_path = "yolo_detection/traffic.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference
    results = model(frame)

    # Draw bounding boxes and count vehicle types
    vehicle_count = {"car": 0, "bike": 0, "bus": 0, "truck": 0}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]


            if label in vehicle_count:
                vehicle_count[label] += 1

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display vehicle count
    print(vehicle_count)

    # Show frame
    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

