import argparse
import csv
import sys

def load_hashes(csv_path, dataset_name):
    """
    Loads hashes from a CSV file into a dictionary.
    Returns: dict {hash: [list_of_image_paths]}
    """
    hash_dict = {}
    print(f"Loading hashes from {dataset_name} ({csv_path})...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header
            
            count = 0
            for row in reader:
                if len(row) < 2:
                    continue
                
                hash_val = row[0]
                image_path = row[1]
                
                if hash_val not in hash_dict:
                    hash_dict[hash_val] = []
                hash_dict[hash_val].append(image_path)
                count += 1
                
        print(f"Loaded {count} entries from {dataset_name}.")
        return hash_dict

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

def find_duplicates(pretrain_csv, eval_csv, output_file):
    # 1. Load Pre-training Hashes
    pretrain_hashes = load_hashes(pretrain_csv, "Pre-training Data")
    
    # 2. Load Evaluation Hashes and check against Pre-training
    eval_hashes = load_hashes(eval_csv, "Evaluation Data")
    
    duplicates = []
    
    print("\nChecking for overlaps...")
    
    # Iterate through evaluation hashes to find matches in pre-training
    for h_val, eval_paths in eval_hashes.items():
        if h_val in pretrain_hashes:
            pretrain_paths = pretrain_hashes[h_val]
            
            # Record every combination of overlapping paths
            for e_path in eval_paths:
                for p_path in pretrain_paths:
                    duplicates.append({
                        'hash': h_val,
                        'eval_image': e_path,
                        'pretrain_image': p_path
                    })

    # 3. Report Results
    if not duplicates:
        print("\n✅ No duplicates found! The datasets appear decontaminated based on pHash.")
    else:
        print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate image pairs!")
        
        if output_file:
            print(f"Saving detailed duplicate report to: {output_file}")
            try:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Hash', 'Evaluation_Image_Path', 'Pretraining_Image_Path'])
                    for item in duplicates:
                        writer.writerow([item['hash'], item['eval_image'], item['pretrain_image']])
                print("Report saved successfully.")
            except Exception as e:
                print(f"Error saving report: {e}")
        else:
            # Print first 10 if no output file specified
            print("First 10 duplicates found:")
            for i, item in enumerate(duplicates[:10]):
                print(f"  {i+1}. Hash: {item['hash']}")
                print(f"     Eval:     {item['eval_image']}")
                print(f"     Pretrain: {item['pretrain_image']}")
            if len(duplicates) > 10:
                print(f"... and {len(duplicates) - 10} more.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for image duplicates between two datasets using pHash CSVs.")
    
    parser.add_argument("pretrain_csv", help="Path to the pre-training hash CSV file")
    parser.add_argument("eval_csv", help="Path to the evaluation hash CSV file")
    parser.add_argument("--output", "-o", default="duplicate_report.csv", help="Output CSV file for the duplicate report (default: duplicate_report.csv)")
    
    args = parser.parse_args()
    
    find_duplicates(args.pretrain_csv, args.eval_csv, args.output)