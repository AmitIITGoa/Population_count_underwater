# Population Count Underwater

## Overview

**Population_count_underwater** is a comprehensive computer vision framework designed for accurate detection, segmentation, and counting of underwater marine life (specifically fish) in challenging underwater environments. 

This project integrates state-of-the-art techniques including:
*   **Image Enhancement**: To improve visibility in turbid underwater footage.
*   **Advanced Detection**: Utilizing YOLOv11 for high-accuracy object detection.
*   **Segmentation**: Leveraging **AquaSAM** (Segment Anything Model) for precise foreground segmentation.
*   **Robust Tracking & Counting**: Custom implementations of **ByteTrack** and **DeepSORT** with specialized logic for occlusion recovery and edge handling to ensure accurate population counts.

## Key Features

*   **Underwater Image Enhancement**: Pre-processing pipeline using deep learning models (UIEC^2-Net, WaterNet) to correct color casts and improve contrast.
*   **High-Performance Detection**: Fine-tuned YOLO models for detecting fish in varied underwater conditions.
*   **Occlusion-Aware Tracking**: Modified ByteTrack algorithm that handles lost tracks due to occlusion or exiting the frame, reducing double counting.
*   **Segmentation Capabilities**: Fine-tuned SAM for generating high-quality masks of underwater creatures.

## Project Structure

```
Population_count_underwater/
├── ADVENCE_DETECTION/          # YOLO-based detection training and inference
│   ├── train_adv.py            # Training script for advanced detection
│   ├── evaluate_metrics.py     # Evaluation scripts
│   └── ...
├── AquaSAM/                    # Underwater Segment Anything Model (SAM)
│   ├── AquaSAM_Inference.py    # Inference script for segmentation
│   ├── train.py                # Fine-tuning script
│   └── ...
├── UWEnhancement/              # Underwater Image Enhancement Toolbox
│   ├── test.py                 # Testing/Inference for enhancement models
│   ├── train.py                # Training enhancement models
│   └── ...
├── Baseline/                   # Baseline models for comparison
├── main_global_feature_edge_bytetrack.py  # Main pipeline using ByteTrack
├── main_global_feature_edge_deepsort.py   # Main pipeline using DeepSORT
└── README.md
```

## Installation

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (recommended for training and real-time inference)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AmitIITGoa/Population_count_underwater.git
    cd Population_count_underwater
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    
    *For general dependencies:*
    ```bash
    pip install numpy opencv-python torch torchvision ultralytics scipy pillow
    ```

    *For specific modules (AquaSAM, UWEnhancement), refer to their respective `README.md` files or `requirements.txt` inside the folders.*

## Usage

### 1. Image Enhancement
Enhance your underwater video or images before detection to improve accuracy.
Navigate to `UWEnhancement/` and run the inference script (refer to `UWEnhancement/README.md` for detailed config).

### 2. Detection & Tracking Pipeline
To run the full counting pipeline on a video file:

**Using ByteTrack (Recommended):**
This script uses YOLO for detection and a custom ByteTrack implementation for tracking.
```bash
python main_global_feature_edge_bytetrack.py
```
*Note: Open the script to modify `VIDEO_PATH`, `OUTPUT_PATH`, and `MODEL_PATH` (path to your trained YOLO weights).*

**Using DeepSORT:**
```bash
python main_global_feature_edge_deepsort.py
```

### 3. Training Detection Models
To train the YOLO model on your own dataset:
```bash
cd ADVENCE_DETECTION
# Edit run.sh or run python directly
python train_adv.py
```

### 4. Segmentation (AquaSAM)
To perform segmentation on underwater images:
```bash
cd AquaSAM
python AquaSAM_Inference.py
```

## Modules Description

### ADVENCE_DETECTION
Contains scripts for training and evaluating advanced object detection models (YOLOv11). It includes scripts for metric evaluation and cluster job submission (`run.sh`).

### AquaSAM
Implementation of **AquaSAM**, an extension of the Segment Anything Model (SAM) tailored for underwater foreground segmentation. It supports fine-tuning on custom underwater datasets.

### UWEnhancement
A toolbox for underwater image enhancement based on PyTorch. It includes implementations of models like UIEC^2-Net and WaterNet to preprocess images for better visibility.

### Tracking Scripts (Root)
*   `main_global_feature_edge_bytetrack.py`: The core counting logic. It implements:
    *   **High/Low Confidence Matching**: To associate weak detections with existing tracks.
    *   **Occlusion Recovery**: Logic to handle fish temporarily hidden behind obstacles.
    *   **Edge Logic**: Prevents ID switching when fish leave and re-enter the frame edges.

## Credits & References

*   **AquaSAM**: [Official Repository](https://github.com/duooppa/AquaSAM)
*   **UWEnhancement**: [Official Repository](https://github.com/BIGWangYuDong/UWEnhancement)
*   **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
*   **ByteTrack**: [Paper/Repo](https://github.com/ifzhang/ByteTrack)

---
*Maintained by AmitIITGoa*