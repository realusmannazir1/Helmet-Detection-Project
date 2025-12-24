# DS-Project — Helmet Detection

A data science and computer vision project for helmet detection using YOLOv8.

## Project Structure

```
DS-Project/
├── Usman Model/                    # Main project folder
│   ├── README.md                   # Detailed setup and usage guide
│   ├── app.py                      # Main application entrypoint
│   ├── run_yolo.py                 # YOLO model demo script
│   ├── onlyvideo.py                # Interactive video detection demo
│   ├── best.pt                     # Trained YOLO model weights
│   ├── helmet.mp4                  # Example video file (optional)
│   ├── Hel.png                     # Screenshot of detection results
│   ├── run_app.bat                 # Windows shortcut to run app.py
│   ├── Requirements                # Python dependencies
│   └── __pycache__/                # Compiled Python files
└── README.md                       # This file
```

## Overview
- Uses Ultralytics YOLO to detect helmets in images/video.
- Includes a demo script `onlyvideo.py` for processing video with interactive controls.
- `app.py` and `run_yolo.py` are other entry points in the project.
- Model weights: `best.pt` (not included in this README; keep it in the same folder).

### Example Detection
![Helmet Detection Screenshot](Usman%20Model/Hel.png)

## Repository structure
- All project files are located in the **`Usman Model/`** folder (see structure above).
- See [Usman Model/README.md](Usman%20Model/README.md) for detailed setup and usage instructions.

## Requirements
- Python 3.10+ recommended
- Virtual environment (recommended)
- Key Python packages (install in your venv):
```bash
pip install -r Requirements
# or, if you don't have a requirements file, at minimum:
pip install ultralytics opencv-python
```

## Setup (Windows)
1. Create and activate a virtual environment (from project root `D:\CODE\DS-Project`):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r "Usman Model\Requirements"  # if you have Requirements file
# or
pip install ultralytics opencv-python
```
3. Ensure `best.pt` and any test media (e.g., `helmet.mp4`) are in `Usman Model` folder.

## Running
- Run the web/app entrypoint (double-clickable on Windows):
  - Double-click `Usman Model\run_app.bat` to activate the venv and run `app.py`.
- Run the video demo in a terminal (recommended from inside the activated venv):
```powershell
cd "Usman Model"
python onlyvideo.py
```

### Controls in `onlyvideo.py`
- SPACE — Play / Pause
- S — Stop (freeze at current frame)
- R — Reset (restart from beginning)
- Q or ESC — Quit
- Close window (X) — script will detect window close and exit cleanly

## Notes and troubleshooting
- If the script raises FileNotFoundError for the model, confirm `best.pt` path and filename match exactly.
- If the video window opens extremely large, the demo creates a resizable window and sets an initial size of 800×600; you can resize or use OS window controls.
- If the script doesn't exit after closing the window, make sure you run it from a terminal where the same Python process is running; closing the GUI window should cause the script to exit automatically.

## Tips
- To process a different video, change the `video_path` variable in `onlyvideo.py`.
- To run `run_yolo.py` from the project root, include the folder in the path or run it from the `Usman Model` directory.

## License
Add your preferred license here, if any.

---
If you want, I can:
- Update the `Requirements` file with exact package versions used,
- Add a `README` section with quick commands to create the virtual env and run common scripts,
- Or add a small help menu printed by `onlyvideo.py` when it starts.
