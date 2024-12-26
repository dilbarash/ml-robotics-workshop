import os
import zipfile
import cv2
import numpy as np
from pathlib import Path

zip_file = 'images.zip'
extract_dir = 'extracted_images'

Path(extract_dir).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

output_dir = 'processed_images'
Path(output_dir).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(extract_dir):
    img_path = os.path.join(extract_dir, filename)
    
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {filename}")
            continue
        
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        img_blurred = cv2.GaussianBlur(img_normalized, (5, 5), 0)
        
        output_img_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(output_img_path, (img_blurred * 255).astype(np.uint8))
        
        print(f"Processed image saved: {output_img_path}")

print("Image preprocessing completed!")
