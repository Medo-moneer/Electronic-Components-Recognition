# AI Project - electronic components Detection

A This project builds an AI system that can detect and identify electronic components (e.g., resistors, capacitors, diodes, etc.) in real-time using a YOLOv8s model.
The system works directly with a webcam, providing live detection and classification of components, and also links detected objects with extra details stored in a CSV metadata file.


## Team Members
1-Mohmmed moneer 202274007
2-Salah aldeen Ahmed 202274008
3-Belal Ahmed 202274035

## Installation and Setup

### Prerequisites
- Python 3.9.13+
- UV package manager (for dependency management)

### Installation Steps
Steps

Clone repository:

git clone https://github.com/your-repo/electro-detect.git
cd electro-detect


Install dependencies:

pip install -r requirements.txt


Train the detection model:

python train_yolo.py


Run real-time detection:

python detect_camera.py---

## Project Structure

```
bottle-label-detection/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── .python-version        # Python version specification
├── main.py                # Real-time detection with YOLO + CNN
├── train.py               # Evaluate model on test dataset
├── esp32/                 # ESP32 microcontroller integration code
├── data/                  # Dataset (train/val/test images)
├── notebooks/             # Jupyter notebooks for experiments
├── Src/                   # contain module data and utlis
└── docs/                  # Documentation and results
```

---

## Usage

Real-Time Detection
python detect_camera.py


Activates live webcam feed

Detects and labels components in real time

Displays bounding boxes + names of components

Pulls component details from Metadata_ElectroCom61.csv and shows them on screen
Model Training
python train_yolo.py


Trains YOLOv8s using dataset defined in data.yaml

Stores weights in runs/detect/train/weights/best.pt

Results

Accuracy: ~92%

Precision: ~91%

Recall: ~95%

F1-Score: ~96%

Findings:

YOLOv8s performs strongly on small, detailed objects like electronic parts.

Metadata integration provides useful descriptive context.

More dataset variety further improves model robustness.---

  

---

Contributing

Fork this repository

Create a feature branch:

git checkout -b Medo-moneer


Implement your changes

Commit and push:

git commit -m "Added new feature"
git push origin electronic components detection


Submit a pull request