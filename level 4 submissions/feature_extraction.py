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

output_dir = 'feature_extracted_images'
Path(output_dir).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(extract_dir):
    img_path = os.path.join(extract_dir, filename)
    
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {filename}")
            continue
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray_img, 100, 200)
        
        corners = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        img[corners > 0.01 * corners.max()] = [0, 0, 255]
        
        output_img_path = os.path.join(output_dir, f"edges_corners_{filename}")
        cv2.imwrite(output_img_path, edges)
        
        print(f"Processed image saved: {output_img_path}")

print("Feature extraction completed!!")
