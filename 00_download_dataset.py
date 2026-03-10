import os
import cv2
import csv
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

# Clean out old data for a fresh run
os.makedirs("data/1_raw_images", exist_ok=True)
os.makedirs("data/2_aligned_faces", exist_ok=True)
os.makedirs("data/3_embeddings", exist_ok=True)
os.makedirs("data/4_database", exist_ok=True)

print("Starting Step 0: Loading LFW Academic Dataset (1000 pairs)...")

# It won't download again; scikit-learn caches it locally.
lfw = fetch_lfw_pairs(subset='train', color=True, slice_=None)

pos_idx = [i for i, t in enumerate(lfw.target) if t == 1][:500]
neg_idx =[i for i, t in enumerate(lfw.target) if t == 0][:500]
selected_indices = pos_idx + neg_idx

csv_data =[]
print("Saving 2000 raw images to disk...")

for i in selected_indices:
    img1 = lfw.pairs[i, 0]
    img2 = lfw.pairs[i, 1]

    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    
    # THE FIX IS HERE: It should be COLOR_RGB2BGR
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    name1 = f"lfw_pair_{i}_A"
    name2 = f"lfw_pair_{i}_B"

    cv2.imwrite(f"data/1_raw_images/{name1}.jpg", img1_bgr)
    cv2.imwrite(f"data/1_raw_images/{name2}.jpg", img2_bgr)

    csv_data.append((name1, name2, lfw.target[i]))

with open("data/test_pairs.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image1", "image2", "label"])
    writer.writerows(csv_data)

print("\nStep 0 Complete! Generated 1000 tests (500 positive, 500 negative).")
print("Saved mapping to 'data/test_pairs.csv'.")
