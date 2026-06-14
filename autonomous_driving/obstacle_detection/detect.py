from ultralytics import YOLO
import cv2
import os

# =========================
# PATHS
# =========================
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE, 'data', 'videos', 'road.mp4')

print("=" * 55)
print(f"Video path : {VIDEO_PATH}")
print(f"Exists     : {os.path.exists(VIDEO_PATH)}")
print("=" * 55)

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video file not found.")
    print("Place a road video at: data/videos/road.mp4")
    exit()

# =========================
# LOAD YOLO MODEL
# =========================
print("\nLoading YOLO model...")
model = YOLO('yolov8n.pt')
print("YOLO loaded successfully!")

# Classes relevant to driving
VEHICLE_CLASSES = [
    'car', 'truck', 'bus',
    'motorbike', 'person', 'bicycle'
]

# =========================
# DETECT OBSTACLES
# =========================
def detect_obstacles(frame):
    results    = model(frame, verbose=False)
    annotated  = frame.copy()
    obstacle_count = 0

    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            confidence = float(box.conf[0])

            if cls_name not in VEHICLE_CLASSES:
                continue
            if confidence < 0.5:
                continue

            obstacle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Red for person, Orange for vehicles
            color = (0, 0, 255) if cls_name == 'person' else (0, 128, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {confidence:.0%}"
            cv2.putText(annotated, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # Show count on screen
    cv2.putText(annotated,
                f"Obstacles detected: {obstacle_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    return annotated, obstacle_count

# =========================
# RUN ON VIDEO
# =========================
print("\nStarting obstacle detection...")
print("Press Q to quit\n")

cap         = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

if not cap.isOpened():
    print("ERROR: Cannot open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1
    output, count = detect_obstacles(frame)

    cv2.putText(output,
                f"Frame: {frame_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow('Obstacle Detection - Press Q to quit', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Processed {frame_count} frames.")
print("Next step: run lane_detection/lane.py")