# Juega Vision – Soccer Video Tracker

![photo](<Screenshot 2025-04-16 at 3.56.43 PM.png>)

Juega Vision is a computer vision app that automatically analyzes soccer videos to detect players and estimate how far they run during a match. Upload a match clip and get tracking + distance stats with a clean browser interface powered by Gradio and YOLOv8.

---

## Features

-  Upload any match clip (.mp4)
-  Automatically detects players and tracks their movements
-  Calculates distance run (in pixels) for each player
-  Displays the processed video with tracking overlays
-  Outputs structured stats (JSON)
-  Browser-based GUI using Gradio

---

### VENV
python3 -m venv venv
source venv/bin/activate

### DEPENDENCIES
pip install ultralytics gradio opencv-python

python app.py

---

## How to Run Locally

clone repo and run app.py!

```bash
git clone https://github.com/your-username/soccer-vision-app.git
cd soccer-vision-app

