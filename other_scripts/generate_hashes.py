import os
import json
import imagehash
from PIL import Image
from tqdm import tqdm
import multiprocessing
import csv

# --- 1. CONFIGURATION ---
META_FILE_PATH = "medconcept_23m.jsonl"
IMAGE_ROOT_DIR = "downloaded_images"
OUTPUT_HASH_MAP_FILE = "pretraining_hash_map_pHash.csv"
NUM_PROCESSES = os.cpu_count() or 4 
CHUNK_SIZE = 1000

# --- 2. WORKER FUNCTION ---
def process_image_chunk(image_paths):
    hash_map = {}
    for relative_path in image_paths:
        try:
            full_image_path = os.path.join(IMAGE_ROOT_DIR, relative_path)
            if not os.path.exists(full_image_path):
                print(f'Image path {full_image_path} does not exist.')
                continue
            
            with Image.open(full_image_path) as img:
                # --- MODIFIED: Using the more robust pHash algorithm ---
                hash_val = str(imagehash.phash(img))
                hash_map[hash_val] = relative_path
        except Exception as e:
            print(f"Error in processing pHash: {e}")
            pass
    print(f"Find unique hashes {len(hash_map)} in this chunk.")
    return hash_map

# --- 3. MAIN SCRIPT ---
def generate_pretraining_hash_map():
    print("Starting pHash map generation for the MedConcept-23M pre-training set...")
    print(f"Using {NUM_PROCESSES} parallel processes.")

    print("Reading image paths from metadata file...")
    all_image_paths = []
    with open(META_FILE_PATH, 'r') as f:
        for line in f:
            try:
                meta = json.loads(line)
                if "image" in meta:
                    all_image_paths.append(meta["image"])
            except json.JSONDecodeError as e:
                print(f'Error: {e}')
                continue

    total_images = len(all_image_paths)
    print(f"Found {total_images} total image entries to process.")
    final_hash_map = {}
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        chunks = [all_image_paths[i:i + CHUNK_SIZE] for i in range(0, total_images, CHUNK_SIZE)]
        for result_map in tqdm(pool.imap_unordered(process_image_chunk, chunks), total=len(chunks), desc="Processing Chunks with pHash"):
            final_hash_map.update(result_map)

    print(f"\nHash generation complete. Found {len(final_hash_map)} unique pHashes.")
    print(f"Saving hash map to {OUTPUT_HASH_MAP_FILE}...")
    with open(OUTPUT_HASH_MAP_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Hash', 'PretrainingImagePath'])
        for hash_val, path in final_hash_map.items():
            writer.writerow([hash_val, path])
    print(f"Successfully saved the complete pre-training pHash map. ✅")

if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None 
    generate_pretraining_hash_map()
