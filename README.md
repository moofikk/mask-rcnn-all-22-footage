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

