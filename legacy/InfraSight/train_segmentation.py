from ultralytics import YOLO

def main():
    print("Loading base YOLOv8 Nano Segmentation model...")
    # This automatically downloads the pre-trained weights to start from
    model = YOLO("yolov8n-seg.pt") 

    print("Starting Training on your RTX 4050...")
    # CRITICAL: Ensure this path correctly points to your data.yaml
    results = model.train(
        data="Pothole-Segmentation-14/data.yaml", 
        epochs=50,          # 50 is perfect for a strong prototype
        imgsz=640,          # The standard size your dataset was resized to
        device=0,           # Forces it to use your RTX 4050 GPU
        project="Pothole_Seg_Project",
        name="Phase1_Master_Model"
    )
    print("Training complete! Check the Pothole_Seg_Project folder for your best.pt file.")

if __name__ == "__main__":
    main()