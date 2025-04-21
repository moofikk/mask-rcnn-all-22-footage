# 🏈 NFL All-22 Game Analyzer

This project is designed to process NFL All-22 football footage using computer vision techniques to detect and track players and field markings like yard lines. The pipeline utilizes **Detectron2** for Mask R-CNN-based instance segmentation and **OpenCV** for line detection and Kalman-filter-based tracking.

---

## 🔧 Features

- Player detection using pretrained Mask R-CNN (`R_50_FPN_3x`) from Detectron2
- Yard line detection using Hough Line Transform
- Line tracking across frames using Kalman Filter
- Frame-by-frame saving and video reconstruction
- Optimized video processing with frame skipping and batch garbage collection

---

## 📁 Project Structure
mask-rcnn-all-22-footage/ ├── main.py # Main script for processing videos ├── requirements.txt # Python dependencies ├── README.md # Project documentation ├── input_videos/ # Folder to place raw All-22 footage ├── output_frames/ # Folder for saving processed frames ├── output_video/ # Folder to store the final video output

---

## ⚙️ Dependencies

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Note: Install Detectron2 according to your system’s CUDA version from the official Detectron2 Installation Guide.

## 🚀 Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/yourusername/nfl-all22-analyzer.git
cd nfl-all22-analyzer
```
2. Install the dependencies (see above).

3. Place your raw All-22 .mp4 footage inside the input_videos/ folder.

4. Run the script:

```bash
python main.py
```

5. Your output frames will be saved in output_frames/ and a compiled video in output_video/output_video.mp4.

## 🔁 Workflow Summary

1. Model Loading: Load pretrained Mask R-CNN from Detectron2’s model zoo.
2. Video Input: Load .mp4 files from input_videos/. - https://youtu.be/1LeQUz8dF60
3. Frame Skipping: Reduce processing time by skipping every n frames.
4. Detection: Apply player segmentation on resized frames.
5. Line Detection: Use Hough Transform and Kalman Filter for yard line tracking.
6. Frame Export: Save visualized detections as .jpg images.
7. Video Compilation: Compile frames into a single .mp4 output. - https://youtu.be/gRzCIjW2y84

## 📌 Important Paths (Editable)
Update the following paths inside main.py to suit your machine:
```bash
input_video_folder = "input_videos"
output_frame_folder = "output_frames"
final_output_path = "output_video/output_video.mp4"
```
## 🧠 Optimizations

Resizing frames to reduce memory load (frame_resize=(640, 360))<br>
Skipping frames using skip_frames=5 to reduce unnecessary computation<br>
Garbage collection (gc.collect()) after processing batches of frames<br>
Batched processing improves performance in long videos<br>

## 🛠️ Future Improvements
Custom segmentation for yard lines and hash marks using fine-tuned Mask R-CNN<br>
Formation recognition and play classification<br>
Integration into a Streamlit dashboard for web-based interaction<br>

## 🧑‍💻 Author
Made by Velan
