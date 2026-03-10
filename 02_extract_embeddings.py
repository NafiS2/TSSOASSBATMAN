import os
import cv2
import numpy as np
from deepface import DeepFace

INPUT_DIR = "data/2_aligned_faces"
OUTPUT_DIR = "data/3_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting Step 2: 512D Feature Extraction (using ArcFace via DeepFace)...")

# We use the 'ArcFace' model specifically to match your assignment logic
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith((".jpg", ".png", ".jpeg")): continue
        
    img_path = os.path.join(INPUT_DIR, filename)
    
    try:
        # DeepFace.represent returns the 512D embedding
        # detector_backend='skip' because the faces are ALREADY cropped/aligned from Step 1
        results = DeepFace.represent(
            img_path = img_path, 
            model_name = "ArcFace", 
            detector_backend = "skip",
            enforce_detection = False
        )
        
        embedding = results[0]["embedding"]
        embedding = np.array(embedding, dtype=np.float32)
        
        # Normalize the vector (Essential for Cosine Similarity!)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        # Save vector as .npy file (remove 'aligned_' prefix from name)
        base_name = filename.replace("aligned_", "").split(".")[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
        np.save(save_path, normalized_embedding)
        
        print(f"Extracted 512D vector for: {base_name}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nStep 2 Complete. Vectors saved to 'data/3_embeddings/'.\n")
