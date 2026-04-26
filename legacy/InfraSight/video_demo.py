import cv2
import torch
from pathlib import Path
from src.utils.logger import setup_logger
from src.models.yolo_segmentation import PotholeSegmenter

logger = setup_logger("VideoDemo")

def main():
    # 1. SETUP
    # Put a test video (e.g., 'road_test.mp4') in your main folder
    input_video_path = "road_test.mp4" 
    output_video_path = "output/presentation_demo.mp4"
    
    Path("output").mkdir(exist_ok=True)

    # 2. LOAD MODEL
    logger.info("Loading Phase 1 Segmentation Engine...")
    # YOLO segmentation models natively calculate bounding boxes too!
    segmenter = PotholeSegmenter("weights/phase1_segmentation_v1.pt", conf_threshold=0.15)

    # 3. INITIALIZE VIDEO STREAMS
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {input_video_path}")
        return

    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    logger.info(f"Processing {total_frames} frames. This will take a few minutes...")

    # 4. THE INFERENCE LOOP
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert BGR (OpenCV) to RGB (AI Model)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference (This handles both boxes and polygon masks)
        results = segmenter.detect(frame_rgb, visualize=True)
        
        # The segmenter wrapper returns the annotated image in RGB
        annotated_frame = results["annotated_image"]
        
        # Convert back to BGR to write to the video file
        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to the final video
        out.write(annotated_bgr)

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            logger.info(f"Processed frame {frame_count}/{total_frames}...")
            
        # CRITICAL FIX FOR YOUR RTX 4050: Clear the VRAM cache so it doesn't crash
        torch.cuda.empty_cache()

    # 5. CLEANUP
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"\nSUCCESS! Presentation video saved to: {output_video_path}")

if __name__ == "__main__":
    main()