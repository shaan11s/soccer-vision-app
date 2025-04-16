from ultralytics import YOLO
import cv2
import math
import os
from collections import defaultdict
import shutil

# Helper to compute Euclidean distance
def pixel_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Load YOLO model once
model = YOLO('model/Final-Model.pt')  # path relative to project root

def run_model(input_path, output_path):
    print(f"\n🚀 Running YOLO model on: {input_path}")

    # Force YOLO to always save to 'runs/track/autotrack'
    results = model.track(
        source=input_path,
        save=True,
        show=False,
        conf=0.3,
        tracker='bytetrack.yaml',  # Adjust if file is elsewhere
        project='runs/track',
        name='autotrack',
        exist_ok=True
    )

    # Expected location of YOLO's saved video
    yolo_output_dir = 'runs/track/autotrack'
    expected_filename = os.path.basename(input_path)
    saved_video_path = os.path.join(yolo_output_dir, expected_filename)

    # Make sure video was created
    if not os.path.exists(saved_video_path):
        print("Output video not found!")
        return "empty"

    # Copy output video to Gradio-accessible folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(saved_video_path, output_path)
    print(f"Copied to: {output_path}")

    # Distance tracking logic 
    track_history = defaultdict(list)
    for result in results:
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes.id)):
                track_id = int(boxes.id[i])
                cls_id = int(boxes.cls[i])
                if cls_id in [0, 1]:  # player or goalkeeper
                    x_center = float(boxes.xywh[i][0])
                    y_center = float(boxes.xywh[i][1])
                    track_history[track_id].append((x_center, y_center))

    # Compute distances
    distance_summary = {}
    for track_id, points in track_history.items():
        distance = sum(pixel_distance(points[i], points[i + 1]) for i in range(len(points) - 1))
        distance_summary[f"Player {track_id}"] = round(distance, 2)

    return {
        "players_detected": len(track_history),
        "distance_run_pixels": distance_summary
    }
