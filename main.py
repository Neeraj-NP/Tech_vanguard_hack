from ultralytics import YOLO
import cv2
import traci
import time

print("Loading YOLO model...")
model = YOLO("yolo_detection/yolov8m.pt")  # Ensure the model exists

print("Checking video file...")
cap = cv2.VideoCapture("yolo_detection/video.mp4")
if not cap.isOpened():
    print("Error: Video file not found or cannot be opened.")
    exit()

sumo_cmd = ["sumo-gui", "-c", "C:/Users/VICTUS/Desktop/Traffic_optimization/sumo_simulation/config.sumocfg"]

print("Starting SUMO with command:", sumo_cmd)
try:
    traci.start(sumo_cmd)
    print("SUMO started successfully!")
except Exception as e:
    print("Error starting SUMO:", e)
    exit()

print("Running SUMO simulation...")
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

traci.close()
print("SUMO simulation completed.")

print("Starting YOLO object detection...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("YOLO detection completed.")
