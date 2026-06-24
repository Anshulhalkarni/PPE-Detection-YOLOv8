# Real-Time Personal Protective Equipment (PPE) Detection Using YOLOv8

An automated, deep learning-based computer vision application designed to monitor industrial and construction work sites for safety compliance. This system utilizes a custom-trained **YOLOv8 (You Only Look Once)** object detection model optimized via NVIDIA CUDA to identify safety gear in live real-time video streams.

---

## 🚀 Key Features
- **Custom-Trained Deep Learning Architecture:** Built on top of the ultra-fast YOLOv8 Nano framework.
- **Hardware-Accelerated Inference:** Native integration with PyTorch and CUDA to leverage local GPU power (tested smoothly on NVIDIA GeForce RTX 3050 Laptop GPU).
- **Real-Time Multi-Object Tracking:** Tracks and identifies multiple object classes simultaneously with an average preprocessing speed of **~5.7ms per frame**.
- **Live Video Interface:** Integrated tracking loop using OpenCV to grab hardware webcam feeds, apply boundary box annotations dynamically, and handle interactive termination.

---

## 🏗️ System Architecture & Workflow

The pipeline runs completely on the local hardware layer, transitioning from raw sensor inputs to annotated frame feeds:

1. **Webcam Feed (OpenCV):** Captures individual frames from the laptop's built-in camera matrix.
2. **GPU Processing Block (PyTorch + CUDA):** Converts frames into tensors and sends them directly to the graphics engine.
3. **Model Evaluation (YOLOv8 Core):** Passes features through the backbone networks to predict bound arrays and class probabilities.
4. **Annotated GUI Window:** Renders real-time bounding boxes around detected safety violations/compliance states.

---

## 📊 Dataset & Model Evaluation
The model was fine-tuned on a multi-class safety dataset containing high-fidelity images of industrial workforce safety gear.

### Model Performance Metrics (After 10 Epochs):
- **mAP50 Accuracy:** `~89.1%` (Mean Average Precision at an intersection-over-union threshold of 0.5)
- **Precision (P):** `82.6%`
- **Recall (R):** `84.3%`
- **Model Storage Footprint:** Core weights file optimized down to a lightweight **6.2 MB** (`best.pt`), making it ideal for edge device deployment.

---

## ⚙️ Local Setup and Installation

Follow these steps to run the live Proof of Concept (POC) pipeline on your local Windows machine:

### 1. Prerequisites
Ensure you have the following installed on your laptop:
- [Python 3.10.x](https://www.python.org/)
- [Git](https://git-scm.com/)
- NVIDIA Graphics Card Drivers (for GPU acceleration)

### 2. Clone the Repository
Open your terminal or PowerShell and run:
```bash
git clone [https://github.com/Anshulhalkarni/PPE-Detection-YOLOv8.git](https://github.com/Anshulhalkarni/PPE-Detection-YOLOv8.git)
cd PPE-Detection-YOLOv8