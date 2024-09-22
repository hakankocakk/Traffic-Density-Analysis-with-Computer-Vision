from ultralytics import YOLO
import torch

def main():

    if torch.cuda.is_available():
        print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU")
    
    model = YOLO('yolov8n.pt')

    data = 'VisDrone.yaml'


    model.train(data=data, epochs=200, batch=16, imgsz=640)

if __name__ == '__main__':
    main()