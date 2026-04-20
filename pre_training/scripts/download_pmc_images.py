import json
import os
import requests
import tarfile
import tempfile
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

# --- Default Configuration ---
# These are used if no command line arguments are provided
DEFAULT_INPUT_FILE = 'medconcept_23m.jsonl'
DEFAULT_OUTPUT_DIR = 'downloaded_images'
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/"

def parse_image_path(image_path_str):
    """
    Parses the 'image' string from JSON to get component parts.
    """
    parts = image_path_str.split('/')
    
    if len(parts) < 4:
        return None, None, None

    dir1, dir2, pmcid, filename = parts[0], parts[1], parts[2], parts[3]
    package_url_suffix = f"{dir1}/{dir2}/{pmcid}.tar.gz"
    local_dir = os.path.join(dir1, dir2, pmcid)
    
    return package_url_suffix, local_dir, filename

def process_meta_file(file_path, limit=None):
    """
    Reads the JSONL and groups required images by their source Package.
    Args:
        file_path: Path to the jsonl file.
        limit: (int) If provided, stops processing after this many lines.
    """
    download_queue = defaultdict(list)
    
    print(f"Reading and parsing {file_path}...")
    if limit:
        print(f"--- DEBUG MODE: Processing only {limit} lines ---")

    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'image' in data:
                    pkg_suffix, local_dir, fname = parse_image_path(data['image'])
                    if pkg_suffix:
                        download_queue[pkg_suffix].append((local_dir, fname))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line.")
                continue
            
            # --- Logic for Limit/Debug ---
            cnt += 1
            if limit is not None and cnt >= limit:
                break
                
    return download_queue

def download_and_process_package(pkg_suffix, images_to_extract, output_root):
    """
    Downloads the tarball, extracts specific files, and saves them locally.
    """
    full_url = BASE_URL + pkg_suffix
    
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        # 1. Download
        try:
            response = requests.get(full_url, stream=True)
            if response.status_code != 200:
                print(f"[Error] Failed to download {full_url} (Status: {response.status_code})")
                return

            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_file.flush()
            temp_file.seek(0)

        except Exception as e:
            print(f"[Error] Connection error for {full_url}: {e}")
            return

        # 2. Extract Specific Files
        try:
            with tarfile.open(fileobj=temp_file, mode='r:gz') as tar:
                for local_dir, target_filename in images_to_extract:
                    pmc_id = os.path.basename(local_dir)
                    internal_path = f"{pmc_id}/{target_filename}"
                    
                    try:
                        member = tar.getmember(internal_path)
                        f_obj = tar.extractfile(member)
                        
                        if f_obj:
                            dest_dir = os.path.join(output_root, local_dir)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, target_filename)
                            
                            with open(dest_path, 'wb') as out_f:
                                shutil.copyfileobj(f_obj, out_f)
                                
                    except KeyError:
                        pass # File not found in tar usually means it's not in the manifest
                    except Exception as e:
                        print(f"[Error] Could not extract {target_filename}: {e}")

        except tarfile.TarError:
            print(f"[Error] Invalid tar file retrieved from {full_url}")

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download images from PMC OA Packages.")
    
    # Input/Output arguments
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default=DEFAULT_INPUT_FILE, 
        help=f'Path to the input JSONL file. Default: {DEFAULT_INPUT_FILE}'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR, 
        help=f'Root directory to save downloaded images. Default: {DEFAULT_OUTPUT_DIR}'
    )
    
    # Debug/Limit arguments
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Run in debug mode (processes first 10 items).'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=None, 
        help='Specific number of items to process (overrides --debug default).'
    )
    
    args = parser.parse_args()

    # Determine the processing limit
    process_limit = None
    if args.limit:
        process_limit = args.limit
    elif args.debug:
        process_limit = 10 

    # --- 2. Check File Existence ---
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # --- 3. Run Logic ---
    print(f"Output Directory: {args.output}")
    queue = process_meta_file(args.input, limit=process_limit)
    print(f"Found {len(queue)} unique packages to process.")

    for pkg_suffix, images_list in tqdm(queue.items(), desc="Downloading Packages"):
        download_and_process_package(pkg_suffix, images_list, args.output)
        
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()