import cv2
from ultralytics import YOLO

# -----------------------------
# Load custom YOLOv8 model
# -----------------------------
model = YOLO("best.pt")

# Flipped class mapping
CLASS_MAP = {
    0: "Helmet",
    1: "No Helmet"
}

# -----------------------------
# Load video
# -----------------------------
video_path = "helmet.mp4"   # 🔹 put your video path here
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps) if fps > 0 else 30

# Create resizable window
window_name = "YOLOv8 Video Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Set initial size

# Control variables
paused = False
stopped = False

print("Controls:")
print("  SPACE - Play/Pause")
print("  S - Stop")
print("  Q - Quit")
print("  F - Fullscreen")
print("  R - Reset Window Size")

# -----------------------------
# Process video
# -----------------------------
while True:
    if not stopped and not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
    
    if not stopped and not paused:
        # YOLO detection
        results = model(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = CLASS_MAP.get(cls_id, "Unknown")
                color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        # Add status text
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show output
    try:
        cv2.imshow(window_name, frame)
    except Exception as e:
        print(f"Window closed: {e}")
        break

    # Keyboard controls
    key = cv2.waitKey(delay) & 0xFF
    
    if key == ord('q'):  # Quit
        print("Quitting...")
        break
    elif key == ord(' '):  # Play/Pause
        paused = not paused
        print(f"{'Paused' if paused else 'Playing'}")
    elif key == ord('s'):  # Stop
        stopped = True
        print("Stopped")
    elif key == ord('r'):  # Reset
        stopped = False
        paused = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Reset")
    elif key == 27:  # ESC key
        print("Closing...")
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
