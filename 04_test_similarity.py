import os
import csv
import numpy as np

print("Starting Step 4: Mass Similarity Testing (1000 Pairs)...")

def load_vec(name):
    # We look for the exact names generated in Step 0
    path = f"data/3_embeddings/{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    return None

def get_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

results =[]
missing_count = 0

# Here is where we read the instructions!
with open("data/test_pairs.csv", "r") as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row
    
    for row in reader:
        # row[0] is image1, row[1] is image2, row[2] is the true label (1 or 0)
        personA, personB, label = row[0], row[1], int(row[2])
        
        v1 = load_vec(personA)
        v2 = load_vec(personB)
        
        # Only compare if MTCNN successfully found a face for BOTH images
        if v1 is not None and v2 is not None:
            score = get_similarity(v1, v2)
            results.append([label, score])
        else:
            missing_count += 1

# Save the mass results for Step 5
np.save("data/test_results.npy", np.array(results))
print(f"Step 4 Complete. Successfully compared {len(results)} pairs.")

if missing_count > 0:
    print(f"Note: {missing_count} pairs were skipped because MTCNN couldn't find a face in the raw image.")
