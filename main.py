---

### 🔄 Updated Python Script (`main.py`)

```python
import sys
import cv2
import os
import gc
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

def load_mask_rcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)

predictor = load_mask_rcnn_model()

kalman_filter = cv2.KalmanFilter(4, 2)
kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

def detect_field_lines(frame):
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return detected_lines

def track_lines(lines):
    tracked_lines = []
    for (x1, y1, x2, y2) in lines:
        measurement = np.array([[np.float32((x1 + x2) / 2)], [np.float32((y1 + y2) / 2)]])
        kalman_filter.correct(measurement)
        prediction = kalman_filter.predict()
        px, py = int(prediction[0]), int(prediction[1])
        tracked_lines.append((px - (x2 - x1) // 2, py - (y2 - y1) // 2, px + (x2 - x1) // 2, py + (y2 - y1) // 2))
    return tracked_lines

def optimized_process_video(video_path, output_dir, predictor, frame_resize=(640, 360), skip_frames=5, batch_size=50):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    if not cap.isOpened():
        print(f"Error: Could not open video {video_name}")
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_idx, batch_counter = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, frame_resize)
        outputs = predictor(frame)

        v = Visualizer(frame[:, :, ::-1], metadata=None, scale=0.8, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        processed_frame = v.get_image()[:, :, ::-1]

        output_frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_frame_path, processed_frame)

        frame_idx += 1
        batch_counter += 1

        if batch_counter >= batch_size:
            gc.collect()
            batch_counter = 0

    cap.release()

def create_video_from_frames(frame_folder, output_video_path, fps=10):
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('jpg')])
    if not frames:
        print("No frames found.")
        return

    first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        image = cv2.imread(os.path.join(frame_folder, frame))
        if image is not None:
            out.write(image)

    out.release()
    cv2.destroyAllWindows()

# === Main Execution ===

input_video_folder = "input_videos"
output_frame_folder = "output_frames"
final_output_path = "output_video/output_video.mp4"

os.makedirs(output_frame_folder, exist_ok=True)
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

for video_file in os.listdir(input_video_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_video_folder, video_file)
        optimized_process_video(video_path, output_frame_folder, predictor)

create_video_from_frames(output_frame_folder, final_output_path)
