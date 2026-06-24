import torch
from ultralytics import YOLO
import os

def main():
    # 1. Path to your best trained weights
    model_path = r"C:\Users\Anshul Halkarni\runs\detect\train-3\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find weights at {model_path}")
        return

    # 2. Load your custom trained YOLOv8 model
    print("Loading custom PPE model...")
    model = YOLO(model_path)

    # 3. Path to an image you want to test
    # (We will use an image from your test split. Change 'test_image.jpg' to a real filename)
    test_image_dir = r"C:\Users\Anshul Halkarni\Downloads\safety_dataset\Personal-Protective-Equipment (PPE) Dataset\test\images"
    
    # Automatically grab the first image in your test folder to make it easy
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print(f"❌ No images found in test directory: {test_image_dir}")
        return
        
    target_image = os.path.join(test_image_dir, test_images[0])
    print(f"Running inference on: {target_image}")

    # 4. Run prediction and save the visual result
    results = model.predict(
        source=target_image,
        conf=0.25,      # Confidence threshold (25%)
        save=True,      # Tells YOLO to save the visual output image with boxes drawn
        line_width=2    # Bounding box line thickness
    )
    
    # 5. Tell the user where to look
    print(f"🎉 Done! Visual results saved inside your project folder under: runs/detect/predict/")

if __name__ == '__main__':
    main()