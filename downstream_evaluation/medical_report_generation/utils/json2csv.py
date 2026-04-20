import json
import csv

# Load the JSON files
with open("../outputs/iu_xray/v1_shallow/conceptclip_l/result/test_refs.json", "r") as refs_file, open("../outputs/iu_xray/v1_shallow/tmpclip_l/result/test_result.json", "r") as result_file:
    gt_data = json.load(refs_file)  # Ground Truth data
    pred_data = json.load(result_file)  # Predicted data

# Prepare data for the CSV file
csv_data = []
for key in gt_data.keys():
    gt_report = gt_data.get(key, [""])[0]  # Get GT report, default to empty string if not found
    pred_report = pred_data.get(key, [""])[0]  # Get Predicted report, default to empty string if not found
    csv_data.append([key, pred_report, gt_report])

# Write to a CSV file
with open("../results/conceptclip_l_iu/conceptclip_l_iu.csv", "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image_ID", "Pred_whole_report", "GT_whole_report"])  # Header
    writer.writerows(csv_data)