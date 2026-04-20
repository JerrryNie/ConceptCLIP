import json
import os
import argparse
from tqdm import tqdm

def minimize_entry(entry):
    """
    Converts a full metadata entry into the minimized format for efficient loading.
    Keeps: caption, image, and simplified umls_meta_info (start/end indices only).
    """
    # Extract simplified umls info: list of [start_char, end_char]
    # We use .get() to handle cases where umls_meta_info might be missing or None
    umls_raw = entry.get('umls_meta_info', [])
    umls_simplified = []
    
    if umls_raw:
        for item in umls_raw:
            # Ensure keys exist before accessing
            if 'start_char' in item and 'end_char' in item:
                umls_simplified.append([item['start_char'], item['end_char']])

    return {
        "caption": entry.get('caption', ""),
        "image": entry.get('image', ""),
        "umls_meta_info": umls_simplified
    }

def main():
    parser = argparse.ArgumentParser(description="Minimize MedConcept-23M metadata for pre-training.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to the original full downloaded jsonl file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./pre_training/src/pretraining_data", 
        help="Directory to save the minimized jsonl file."
    )
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_filename = os.path.basename(args.input_path)
    output_filename = input_filename.replace('.jsonl', '_minimized.jsonl')
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Processing: {args.input_path}")
    print(f"Target: {output_path}")

    # Count total lines for tqdm (optional but helpful for large files)
    # This might take a moment for 45GB, so we can skip counting or do a quick estimate.
    # For a generic script, we'll iterate directly to save startup time, 
    # but using tqdm on the file object provides a speed indicator (it/s).

    with open(args.input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Minimizing records"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                minimized_data = minimize_entry(data)
                f_out.write(json.dumps(minimized_data) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line.")
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"\nSuccess! Minimized file saved to: {output_path}")

if __name__ == "__main__":
    main()