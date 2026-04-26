import os
import cv2
import random
from pathlib import Path
import numpy as np

def prepare_dataset():
    # Paths
    base_dir = Path(__file__).parent.parent.parent.absolute()
    output_dir = base_dir / "data/processed/road_material_classification"
    
    # Texture paths (from artifacts)
    concrete_tex = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\95d3c7f0-175d-4a96-8f00-82c826854c04\concrete_road_texture_1_1772420387625.png")
    paving_tex = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\95d3c7f0-175d-4a96-8f00-82c826854c04\paving_road_texture_1_1772420403973.png")
    
    # Asphalt source (RDD2022)
    asphalt_source_dir = base_dir / "data/raw/RDD2022/India/train/images"
    
    # Create structure
    for split in ['train', 'val']:
        for cls in ['asphalt', 'concrete', 'paving']:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print("--- Preparing Asphalt patches from RDD2022 ---")
    asphalt_images = list(asphalt_source_dir.glob("*.jpg"))[:200]
    for i, img_path in enumerate(asphalt_images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape
        # Take a patch from the bottom 1/3 (usually road)
        if h > 300 and w > 300:
            patch = img[h-250:h-26, w//2-112:w//2+112]
            split = 'train' if i < 160 else 'val'
            cv2.imwrite(str(output_dir / split / 'asphalt' / f"asphalt_{i}.jpg"), patch)

    print("--- Preparing Concrete patches from textures ---")
    con_img = cv2.imread(str(concrete_tex))
    if con_img is not None:
        count = 0
        h, w, _ = con_img.shape
        for r in range(0, h-224, 200):
            for c in range(0, w-224, 200):
                patch = con_img[r:r+224, c:c+224]
                # Add some noise/variation
                noise = np.random.normal(0, 5, patch.shape).astype(np.uint8)
                patch = cv2.add(patch, noise)
                split = 'train' if count % 5 != 0 else 'val'
                cv2.imwrite(str(output_dir / split / 'concrete' / f"concrete_{count}.jpg"), patch)
                count += 1
                if count > 200: break
            if count > 200: break

    print("--- Preparing Paving patches from textures ---")
    pav_img = cv2.imread(str(paving_tex))
    if pav_img is not None:
        count = 0
        h, w, _ = pav_img.shape
        for r in range(0, h-224, 200):
            for c in range(0, w-224, 200):
                patch = pav_img[r:r+224, c:c+224]
                split = 'train' if count % 5 != 0 else 'val'
                cv2.imwrite(str(output_dir / split / 'paving' / f"paving_{count}.jpg"), patch)
                count += 1
                if count > 200: break
            if count > 200: break

    print(f"Preparation done! Data saved to {output_dir}")

if __name__ == "__main__":
    prepare_dataset()
