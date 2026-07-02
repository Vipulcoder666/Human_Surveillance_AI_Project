import cv2
import time
from ultralytics import YOLO

# =====================================
# Load YOLO Model
# =====================================
model = YOLO("yolo11m.pt")  # Accuracy > Speed

# =====================================
# RTSP Camera URL
# =====================================
camera_url = "rtsp://admin:cctv@321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"

cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        break

    # ---------------------------------
    # Human Detection
    # ---------------------------------
    results = model(
        frame,
        classes=[0],      # Person Class
        conf=0.45,
        verbose=False
    )

    annotated_frame = frame.copy()

    person_count = 0

    for result in results:

        for box in result.boxes:

            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Bounding Box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Person Number + Confidence
            cv2.putText(
                annotated_frame,
                f"P{person_count} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ---------------------------------
    # FPS
    # ---------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # FPS Display
    cv2.putText(
        annotated_frame,
        f"FPS : {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Person Count Display
    cv2.putText(
        annotated_frame,
        f"Persons : {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ---------------------------------
    # Show Output
    # ---------------------------------
    cv2.imshow("Human Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =====================================
# Cleanup
# =====================================
cap.release()
cv2.destroyAllWindows()