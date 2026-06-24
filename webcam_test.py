import torch
from ultralytics import YOLO
import cv2
import os

def main():
    # Points cleanly to the model file right inside your project directory
    model_path = os.path.join("models", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Cannot find the model at: {model_path}")
        return
        
    print(f"🚀 Loading custom PPE model from project files: {model_path}")
    model = YOLO(model_path)

    # Open the laptop webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🚀 Live Webcam Started!")
    print("Click on the video window and press 'q' to EXIT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, device=0, conf=0.25, stream=True)

        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()
            break 

        cv2.imshow("Custom PPE Live Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()