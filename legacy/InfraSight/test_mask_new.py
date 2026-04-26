from ultralytics import YOLO

def main():
    print("Loading your custom Phase 1 Segmentation Engine...")
    # Ensure this path points exactly to your trained weights
    model = YOLO("runs/segment/Pothole_Seg_Project/Phase1_Master_Model/weights/best.pt")

    # Define the image you want to test
    test_image = "test8.png"  # Change this to whatever image you are testing

    print(f"\nRunning inference on {test_image} with conf=0.10...")
    
    # We lower the confidence threshold from the default 0.25 to 0.10
    # This forces the model to show masks for shallower/fainter potholes
    # save=True will save the output image into the 'runs/segment/predict' folder
    results = model.predict(
        source=test_image, 
        conf=0.10,      # The critical adjustment
        iou=0.45,       # Prevents overlapping masks
        show=True,      # Pops open a window to show you the result immediately
        save=True
    )

    print("\nInference complete! Check the pop-up window or the 'runs' folder.")

if __name__ == "__main__":
    main()