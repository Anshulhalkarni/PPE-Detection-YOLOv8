import torch
from ultralytics import YOLO

def main():
    # 1. Double check if CUDA is actually available on your laptop system
    if torch.cuda.is_available():
        print(f"🚀 CUDA is available! Training will run on: {torch.cuda.get_device_name(0)}")
        selected_device = 0  # Force CUDA GPU execution
    else:
        print("⚠️ WARNING: CUDA is not detected by PyTorch. Falling back to CPU.")
        selected_device = "cpu"

    # 2. Load the pre-trained small YOLO model baseline
    model = YOLO("yolov8n.pt")

    # 3. Fire up the training pipeline on your Graphics Card
    results = model.train(
        data="data.yaml",     
        epochs=10,            
        imgsz=640,            
        device=selected_device, # Dynamically locks onto your CUDA GPU
        workers=2             
    )

if __name__ == '__main__':
    main()