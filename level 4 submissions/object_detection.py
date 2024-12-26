import os
import zipfile
import cv2
from pathlib import Path

zip_file = 'images.zip'
extract_dir = 'extracted_images'

Path(extract_dir).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

output_dir = 'object_detected_images'
Path(output_dir).mkdir(parents=True, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(extract_dir):
    img_path = os.path.join(extract_dir, filename)
    
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {filename}")
            continue
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        output_img_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_img_path, img)
        
        print(f"Processed image saved: {output_img_path}")

print("Object detection completed!!!")
