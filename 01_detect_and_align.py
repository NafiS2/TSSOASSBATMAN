import os
import cv2
import math
from mtcnn import MTCNN

# Disable the annoying TensorFlow info logs to keep your terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

detector = MTCNN()
INPUT_DIR = "data/1_raw_images"
OUTPUT_DIR = "data/2_aligned_faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nStarting Step 1: Face Detection & Alignment...")

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith((".jpg", ".png", ".jpeg")): continue
        
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = detector.detect_faces(img_rgb)
    if not results:
        print(f"Skipping {filename} - No face detected.")
        continue
        
    # Get main face and align
    face = max(results, key=lambda b: b['confidence'])
    left_eye, right_eye = face['keypoints']['left_eye'], face['keypoints']['right_eye']
    
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    
    # THE FIX: Force standard Python floats so OpenCV doesn't crash
    center_x = float((left_eye[0] + right_eye[0]) / 2.0)
    center_y = float((left_eye[1] + right_eye[1]) / 2.0)
    eyes_center = (center_x, center_y)
    
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_img = cv2.warpAffine(img_rgb, M, (img.shape[1], img.shape[0]))
    
    x, y, w, h = face['box']
    
    # Safely crop the image, preventing negative indices if the face is near the edge
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    
    cropped = aligned_img[y1:y2, x1:x2]
    
    if cropped.size == 0:
        print(f"Skipping {filename} - Crop failed due to edge boundaries.")
        continue
        
    final_face = cv2.resize(cropped, (112, 112))
    
    # Save aligned face for the next script
    save_path = os.path.join(OUTPUT_DIR, f"aligned_{filename}")
    cv2.imwrite(save_path, cv2.cvtColor(final_face, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved: {save_path}")

print("Step 1 Complete. Check the 'data/2_aligned_faces' folder.\n")
