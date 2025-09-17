Model Performance Documentation
Overview

This document summarizes the performance metrics and results of an AI Project – Electronic Components Detection.
The goal of the project is to identify and classify different types of electronic components from live camera input.

The trained model is integrated with a computer vision pipeline to display results in real-time:

✅ Correctly recognized component → Name and confidence displayed on screen.

❌ Unrecognized or low-confidence input → No label shown.

Model Details

Algorithm: Convolutional Neural Network (ResNet18)

Framework: PyTorch

Input: Images of electronic components captured via webcam.

Dataset Split

Training: 80%

Validation: 20%

Testing: 30%

Epochs: 15

Batch Size: 35

Preprocessing & Features

The model was trained directly on image data. Each image was preprocessed as follows:

Resizing: Images resized to 224×224 pixels.

Normalization: Pixel values normalized using ImageNet mean & std.

Augmentation: Random rotations and horizontal flips applied to improve generalization.

Performance Metrics

Accuracy: 91.3%

Precision (Average across components): 90.7%

Recall (Average across components): 92.1%

F1-Score (Average across components): 91.4%

(Values are illustrative, replace with your actual evaluation results)

Key Findings

ResNet18 performed well in distinguishing between multiple component classes.

Data augmentation significantly reduced overfitting on the training set.

Real-time classification was achieved by selecting a Region of Interest (ROI) in the camera feed.

Predictions include both the class name and the confidence score shown directly on the video stream.

Recommendations

Collect a larger dataset of electronic components under varied lighting and angles.

Experiment with object detection models (e.g., YOLOv8s) for automatic detection without manual ROI selection.

Explore quantization or lightweight models for deployment on edge devices.

Add multi-language support for displaying component names.