import os
import numpy as np
import faiss
import json

INPUT_DIR = "data/3_embeddings"
DB_DIR = "data/4_database"
os.makedirs(DB_DIR, exist_ok=True)

print("Starting Step 3: Building FAISS O(N) Database...")

dimension = 512
index = faiss.IndexFlatIP(dimension) # Inner Product == Cosine Similarity
metadata = {}

vector_id = 0
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".npy"): continue
        
    # Load vector
    vector = np.load(os.path.join(INPUT_DIR, filename))
    vector = np.array([vector], dtype=np.float32)
    
    # Add to FAISS index
    index.add(vector)
    
    # Store the person's name mapping
    person_name = filename.replace(".npy", "")
    metadata[vector_id] = person_name
    vector_id += 1

# Save Database to disk
faiss.write_index(index, os.path.join(DB_DIR, "face_index.bin"))
with open(os.path.join(DB_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"Step 3 Complete. Indexed {vector_id} faces into FAISS database.\n")
