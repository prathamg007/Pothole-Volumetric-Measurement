from ultralytics import YOLO

def main():
    # Load YOUR custom trained weights
    # Make sure this path exactly matches where your best.pt is saved
    model = YOLO("runs/segment/Pothole_Seg_Project/Phase1_Master_Model/weights/best.pt")

    # Run the model on your test image
    print("Running segmentation engine...")
    results = model.predict(source="test9.png", show=True, save=True)

    print("\nCheck the 'runs/segment/predict' folder to see the drawn masks!")

if __name__ == "__main__":
    main()