# 🐟 Population Count Underwater

> **A robust computer vision framework for detecting, tracking, and counting marine life in challenging underwater environments.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv11](https://img.shields.io/badge/YOLO-v11l-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

**Population_count_underwater** is an advanced system designed to automate the census of fish populations. Unlike standard tracking pipelines, this project addresses specific underwater challenges such as **turbidity**, **occlusion**, and **erratic movement**.

We combine **Image Enhancement**, **YOLOv11 Detection**, **AquaSAM Segmentation**, and a **Custom ByteTrack** algorithm to achieve high-accuracy counting.

---

## ✨ Key Features

*   **🌊 Underwater Image Enhancement**: Pre-processing with **UIEC^2-Net** and **WaterNet** to restore color and clarity.
*   **🎯 Advanced Detection**: Powered by **YOLOv11l (Large)**, fine-tuned for underwater species detection.
*   **🧠 Intelligent Tracking (Custom ByteTrack)**:
    *   **Occlusion Recovery**: Re-identifies fish that temporarily disappear behind rocks or plants.
    *   **Edge Logic**: Prevents double-counting when fish exit and re-enter the frame boundaries.
    *   **Proximity Boost**: Uses spatial locality to maintain track identity in low-visibility conditions.
*   **🎨 Precision Segmentation**: Uses **AquaSAM** (Segment Anything Model) for pixel-perfect foreground extraction.

---

## 📂 Project Structure

```plaintext
Population_count_underwater/
├── 📁 ADVENCE_DETECTION/          # YOLOv11 Training & Evaluation
│   ├── train_adv.py               # Main training script
│   ├── evaluate_metrics.py        # Performance metrics
│   └── predictions/               # Detection results (JSON/Images)
│
├── 📁 AquaSAM/                    # Segmentation Module
│   ├── AquaSAM_Inference.py       # Run segmentation on images
│   └── train.py                   # Fine-tune SAM
│
├── 📁 UWEnhancement/              # Image Enhancement Toolbox
│   ├── test.py                    # Enhancement inference
│   └── config/                    # Model configurations (WaterNet, etc.)
│
├── 📁 Baseline/                   # Comparison models
│
├── main_global_feature_edge_bytetrack.py  # Alternative Pipeline (without Re-identification)(ByteTrack)
├── main_global_feature_edge_deepsort.py   #  🚀 MAIN PIPELINE (DeepSORT)
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/AmitIITGoa/Population_count_underwater.git
cd Population_count_underwater

# Create a virtual environment (Recommended)
conda create -n fish_count python=3.9
conda activate fish_count

# Install dependencies
pip install numpy opencv-python torch torchvision ultralytics scipy pillow matplotlib
```

### 2. Step 1: Detection (Training & Setup)

Before tracking, you need a robust detection model. The **ADVENCE_DETECTION** folder contains the tools to train and evaluate YOLOv11 on your underwater dataset.

**To Train the Model:**
```bash
cd ADVENCE_DETECTION
python train_adv.py
```
*   **Output**: This will generate a `best.pt` weight file in `ADVENCE_DETECTION/runs/detect/train/weights/`.
*   **Note**: You can skip this if you already have a pre-trained model.

📥 **Download Pre-trained Models:**
You can download our best trained models (for Detection & DeepSORT) from this [Google Drive Link](https://drive.google.com/drive/folders/1rxQrHpoiZW1e_Tczec0YUYBWQ0xkZuBT?usp=sharing).

### 3. Step 2: Tracking & Counting (Main Pipeline)

Once you have your detection model (`best.pt`), run the tracking pipeline. We recommend using **DeepSORT** for the most accurate population counting.

**Using DeepSORT (Recommended):**
This script integrates YOLO detection with DeepSORT tracking (using Re-ID features) to count fish accurately.

```bash
python main_global_feature_edge_deepsort.py
```

**Configuration:**
Open `main_global_feature_edge_deepsort.py` to adjust paths:
```python
VIDEO_PATH = "output_video2_enhanced.mp4"       # Input video
OUTPUT_PATH = "output_video2_deepsort.mp4"      # Output video
STATS_OUTPUT = "population_statistics.txt"      # Count statistics
MODEL_PATH = "best.pt"                          # Path to your trained YOLO weights
```

**Alternative: ByteTrack**
If you prefer the motion-based tracker:
```bash
python main_global_feature_edge_bytetrack.py
```

### 4. Running Segmentation (AquaSAM)

To generate segmentation masks for analysis:

```bash
cd AquaSAM
python AquaSAM_Inference.py
```
*   This uses the fine-tuned SAM model to segment fish from the background.

---

## 🧠 Methodology Details

### ⚙️ Pipeline Overview
The system follows a strict two-stage process:
1.  **Detection**: First, the YOLOv11 model detects all fish in the current frame.
2.  **Tracking**: These detections are then passed to the tracking algorithm to associate them with existing IDs.

### 🔍 Detection Method: YOLOv11
We utilize the **YOLOv11 Large** model for its superior balance of speed and accuracy.
*   **Input**: Enhanced underwater frames.
*   **Output**: Bounding boxes with confidence scores.
*   **Location**: `ADVENCE_DETECTION/`

### 🛤️ Tracking Methods
We provide two tracking implementations. **DeepSORT is currently recommended** for its robustness in re-identifying fish after occlusion.

#### 1. DeepSORT (Recommended - Re-ID Based)
*   **Mechanism**: Uses a deep learning model to extract appearance features (Re-ID) from bounding boxes.
*   **Pros**: Excellent at handling long-term occlusions and re-identifying fish that leave and re-enter the frame, provided they look distinct.
*   **Cons**: Slightly more computationally intensive than ByteTrack.

#### 2. ByteTrack (Motion Based - No Re-ID)
*   **Mechanism**: Relies primarily on high-performance motion prediction (Kalman Filter) and Intersection over Union (IoU). **Does not perform Re-ID feature extraction.**
*   **Key Advantage**: Associates low-confidence detections that other trackers ignore.
*   **Use Case**: Best when fish look identical and motion is the only reliable differentiator.

---

## 📊 Results & Outputs

*   **Video Output**: Located in the root directory (e.g., `output_video2_bytetrack_fixed_v3.mp4`).
*   **Population Stats**: Text files (e.g., `population_statistics_v7.txt`) containing the final count.
*   **Training Metrics**: Located in `ADVENCE_DETECTION/runs/detect/`.
*   **Predictions**: JSON format results in `ADVENCE_DETECTION/predictions/`.

---

## 👥 Credits

*   **Maintainer**: AmitIITGoa
*   **References**:
    *   [AquaSAM](https://github.com/duooppa/AquaSAM)
    *   [ByteTrack](https://github.com/ifzhang/ByteTrack)
    *   [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
    *   [UWEnhancement](https://github.com/BIGWangYuDong/UWEnhancement)
