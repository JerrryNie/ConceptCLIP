# Medical Report Generation
This repository contains the code for the downstream medical report generation task based on ConceptCLIP. The minimum requirement to run the code is a 24 GB 3090 Ti GPU.

## Installation
- Install basic requirements
```bash
pip install -r requirements.txt
```
- Download [pycocoevalcap](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ynieae_connect_ust_hk/EbyNewnEQCtKjnW8C8oJl3oBDwIUsyeUE8R1AnMKD_VKjA?e=fAxiGZ), move the zip file into `downstream_evaluation/medical_report_generation/utils` and unzip it.


## Data Preparation
The IU-Xray dataset can be downloaded [here](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) and move into the folder `dataset`.

## Model Checkpoints
- **Base LLM:** [Llama‑2‑7b‑chat‑hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). Download it and move it to the folder `models/ckpt`.  
- **Pre‑trained & auxiliary checkpoints:** [ConceptCLIP](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ynieae_connect_ust_hk/EQ0QEuUIqZFJkjAwz0GzcSABQJ51zyBoofMvOM34--D3bQ?e=CBFFgL). Download it and move it to the folder `models/ckpt`.
- **CheXbert:** [CheXbert](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ynieae_connect_ust_hk/EbudM5GGasFJjz5-H0KglmkB0bNHw29UMmAEUj-2hhJ4mQ?e=2FgYft). Download it and move it to the folder `utils`.

## Inference
After the data and checkpoints are ready, you can inference the model on **IU-Xray** with our pre‑trained weights.
1. Modify the paths in the `scripts/1-2.shallow_test_iuxray_conceptclip_l` file according to your environment settings.
2. Run the `scripts/1-2.shallow_test_iuxray_conceptclip_l` script.
```bash
cd scripts
bash 1-2.shallow_test_iuxray_conceptclip_l.sh
```
The inference results will be saved in the `$savepath/result/test_refs.json` file.

## Evaluation
Next, you can evaluate the inference results on the **IU-Xray** dataset using the evaluation tools provided in the `utils` directory.
> Remember to modify the paths in these files according to your environment settings.
1. Generating the results csv file.
```bash
cd utils
python json2csv.py
```
2. Compute Natural Language Generation (NLG) metrics
```bash
python pycoco_nlg_evaluation.py
```
3. Compute Clinical Efficacy (CE) metrics
```bash
python chexbert_ce_evaluation.py
```

## Training
You can also train the model on **IU-Xray** dataset.
1. Modify the paths in the `scripts/1-1.shallow_train_iuxray_conceptclip_l` file according to your environment settings.
2. Run the `scripts/1-1.shallow_train_iuxray_conceptclip_l` script.
```bash
cd scripts
bash 1-1.shallow_train_iuxray_conceptclip_l.sh
```
The training logs will be saved in the `scripts/logs` directory.