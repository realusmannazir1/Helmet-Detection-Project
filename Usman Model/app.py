import cv2
import os
import tkinter as tk
from tkinter import Label, Button, Frame, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# -----------------------------
# Load CUSTOM YOLOv8 model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Flipped class labels
# -----------------------------
CLASS_MAP = {
    1: "Helmet",
    0: "No Helmet"
}

# -----------------------------
# App Window
# -----------------------------
app = tk.Tk()
app.title("Helmet Detection - YOLOv8")
app.geometry("1200x700")
app.columnconfigure(1, weight=1)
app.rowconfigure(0, weight=1)

# -----------------------------
# Left Panel
# -----------------------------
button_frame = Frame(app, bg="#222")
button_frame.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

# -----------------------------
# Camera Display
# -----------------------------
camera_label = Label(app, bg="black")
camera_label.grid(row=0, column=1, sticky="nsew")

status_label = Label(app, text="", bg="#222", fg="white", font=("Arial", 12))
status_label.grid(row=1, column=1, sticky="ew", padx=5, pady=(0,5))

# -----------------------------
# Globals
# -----------------------------
cap = None
running = False
is_file_source = False
video_path = None

# -----------------------------
# Functions
# -----------------------------

def show_status(msg, color="white"):
    try:
        status_label.config(text=msg, fg=color)
    except Exception:
        pass


def load_video():
    global video_path, is_file_source
    path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
    )
    if not path:
        return
    video_path = path
    is_file_source = True
    try:
        loaded_label.config(text=os.path.basename(video_path))
    except Exception:
        pass
    show_status(f"Video loaded: {os.path.basename(video_path)}", "lightgreen")


def start_video(path):
    global cap, running, is_file_source, video_path
    if running:
        return
    if not path:
        show_status("No video loaded", "red")
        return
    if cap:
        cap.release()
    cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        show_status("Failed to open video.", "red")
        if cap:
            cap.release()
        return
    is_file_source = True
    running = True
    show_status("Video started", "lightgreen")
    process_frame()


def start_camera():
    global cap, running, is_file_source
    if running:
        return
    if cap:
        cap.release()
    cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        show_status("Failed to open camera.", "red")
        if cap:
            cap.release()
        return
    is_file_source = False
    running = True
    show_status("Camera started", "lightgreen")
    process_frame()


def stop_camera():
    global running, cap, is_file_source, video_path
    running = False
    if cap:
        cap.release()
    camera_label.config(image="")
    is_file_source = False
    video_path = None
    try:
        loaded_label.config(text="No video loaded")
    except Exception:
        pass
    show_status("Stopped", "white")


def process_frame():
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        # if file source, it means video ended -> stop
        if is_file_source:
            show_status("Video ended", "white")
            stop_camera()
            return
        # camera: retry
        camera_label.after(100, process_frame)
        return

    if not is_file_source:
        frame = cv2.flip(frame, 1)

    # ------------------ YOLO DETECTION ------------------
    try:
        results = model(frame, conf=0.5, verbose=False)
    except Exception as e:
        show_status(f"Inference error: {e}", "red")
        camera_label.after(500, process_frame)
        return
    else:
        show_status("")

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            # extract coords
            try:
                xy = box.xyxy
                if hasattr(xy, "cpu"):
                    coords = xy.cpu().numpy().flatten()
                elif hasattr(xy, "tolist"):
                    coords = xy.tolist()
                else:
                    coords = list(xy)
                if len(coords) >= 4:
                    x1, y1, x2, y2 = map(int, coords[:4])
                else:
                    continue
            except Exception:
                continue

            # class id
            cls_id = None
            try:
                cls_val = box.cls
                if hasattr(cls_val, "item"):
                    cls_id = int(cls_val.item())
                elif hasattr(cls_val, "__len__"):
                    cls_id = int(cls_val[0])
                else:
                    cls_id = int(cls_val)
            except Exception:
                cls_id = None

            # confidence
            try:
                conf_val = box.conf
                if hasattr(conf_val, "item"):
                    conf = float(conf_val.item())
                elif hasattr(conf_val, "__len__"):
                    conf = float(conf_val[0])
                else:
                    conf = float(conf_val)
            except Exception:
                conf = 0.0

            # label: prefer model.names
            if hasattr(model, "names") and cls_id in model.names:
                label = model.names[cls_id]
            else:
                label = CLASS_MAP.get(cls_id, "Unknown")

            color = (0, 255, 0) if "helmet" in str(label).lower() else (0, 0, 255)

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

    # ------------------ SHOW IN TKINTER ------------------
    w = camera_label.winfo_width()
    h = camera_label.winfo_height()
    if w > 1 and h > 1:
        frame = cv2.resize(frame, (w, h))

    img = ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    camera_label.imgtk = img
    camera_label.configure(image=img)

    camera_label.after(10, process_frame)

# -----------------------------
# Buttons
# -----------------------------
loaded_label = Label(button_frame, text="No video loaded", bg="#222", fg="white", font=("Arial", 10))
loaded_label.pack(pady=(10,4))

load_btn = Button(button_frame, text="Load Video",
                  font=("Arial", 12), bg="#0055aa", fg="white",
                  width=18, height=1, command=lambda: load_video())
load_btn.pack(pady=(0,8))

start_btn = Button(button_frame, text="Start Camera",
                   font=("Arial", 14), bg="#008000", fg="white",
                   width=18, height=2, command=start_camera)
start_btn.pack(pady=10)

start_video_btn = Button(button_frame, text="Start Video",
                         font=("Arial", 14), bg="#006600", fg="white",
                         width=18, height=2, command=lambda: start_video(video_path))
start_video_btn.pack(pady=10)

stop_btn = Button(button_frame, text="Stop",
                  font=("Arial", 14), bg="#444", fg="white",
                  width=18, height=2, command=stop_camera)
stop_btn.pack(pady=10)

quit_btn = Button(button_frame, text="Quit",
                  font=("Arial", 14), bg="#aa0000", fg="white",
                  width=18, height=2, command=app.destroy)
quit_btn.pack(pady=10) 

# -----------------------------
# Run App
# -----------------------------
app.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()
