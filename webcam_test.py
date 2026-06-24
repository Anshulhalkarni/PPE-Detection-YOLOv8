import torch
from ultralytics import YOLO
import cv2
import os

def main():
    # Target the exact restored folder path
    model_path = r"C:\Users\Anshul Halkarni\runs\detect\train-3\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Still cannot find the model at: {model_path}")
        print("Please check your File Explorer and see if the folder name is 'train', 'train-2', or 'train-3'.")
        return
        
    print(f"🚀 Found your restored weights at: {model_path}")
    print("Loading custom PPE model onto GPU...")
    model = YOLO(model_path)

    # Open the laptop webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🚀 Live Webcam Started!")
    print("👉 IMPORTANT: Click onto the video window screen and press 'q' to EXIT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run live tracking on your RTX 3050 (device=0)
        results = model.track(source=frame, persist=True, device=0, conf=0.25, stream=True)

        annotated_frame = frame  
        for r in results:
            annotated_frame = r.plot()
            break  

        # Display the live window
        cv2.imshow("Custom PPE Live Detector", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()