# Medical Report Generation
This repository contains the code for the downstream medical report generation task based on ConceptCLIP. The minimum requirement to run the code is a 24 GB 3090 Ti GPU.

## Installation
Run the setup commands from `downstream_evaluation/medical_report_generation`.

- Install basic requirements:

```bash
pip install -r requirements.txt
```
- Install the caption evaluation tools from the [upstream pycocoevalcap project](https://github.com/salaniz/pycocoevalcap):

```bash
pip install pycocoevalcap
```

The standalone NLG evaluator imports `pycocoevalcap`. The report-generation model currently imports the scorers under the legacy name `evalcap`; to supply that module from the same upstream source, clone it into this task directory:

```bash
git clone https://github.com/salaniz/pycocoevalcap.git evalcap
```

Follow the upstream Java requirements for METEOR. The former personal evaluation-tool archive is no longer used.

## Data Preparation

Obtain the Indiana University chest X-ray images and reports directly from [NLM Open-i](https://openi.nlm.nih.gov/) using its [official collection/download FAQ](https://openi.nlm.nih.gov/faq). Follow NLM's access and reuse terms and cite the original dataset. We no longer direct users to a third-party Kaggle copy. If Open-i is temporarily under maintenance, retry later or contact NLM.

The supplied [dataset/annotation.json](./dataset/annotation.json) defines the report-generation splits and image paths. Prepare your authorized local images under `dataset/images` to match each record's `image_path`, for example `dataset/images/CXR2384_IM-0942/0.png` and `1.png`. The upstream archive may use different filenames; use the original image/report associations to map them to the annotation entries. This repository does not include an automatic conversion from the original release to that layout. Set `annotation` and `base_dir` in the run scripts if you use different local paths.

## Model Checkpoints

- **Base LLM:** Obtain [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) from Meta's Hugging Face release, completing its access process. Save it to `models/ckpt/Llama-2-7b-chat-hf`, or set `--llama_model` to your local path.
- **ConceptCLIP backbone:** Obtain the model from the [official ConceptCLIP release](https://huggingface.co/JerrryNie/ConceptCLIP), after requesting access and authenticating. Update the local `AutoModel.from_pretrained(...)` path in [models/R2GenGPT_ConceptCLIP_L.py](./models/R2GenGPT_ConceptCLIP_L.py) to your downloaded model directory; the current path is relative to the working directory.
- **Report-generation checkpoint:** Inference additionally requires a task-trained checkpoint selected by `delta_file` in [scripts/1-2.shallow_test_iuxray_conceptclip_l.sh](./scripts/1-2.shallow_test_iuxray_conceptclip_l.sh), currently `models/ckpt/conceptclip_l_iu.pth`. The backbone release is not a substitute for this checkpoint. The former shared checkpoint archive is unavailable, and no replacement download for the task checkpoint is linked here. Train the task using the instructions below and point `delta_file` to the resulting checkpoint, or use a compatible checkpoint you already have.
- **CheXbert:** Follow the [official Stanford CheXbert checkpoint instructions](https://github.com/stanfordmlgroup/CheXbert#checkpoint-download) and its license. Save the checkpoint as `utils/chexbert.pth`, or update the path in [utils/chexbert_ce_evaluation.py](./utils/chexbert_ce_evaluation.py). Record the checkpoint version when reporting CE metrics; the current upstream release is not guaranteed to match the retired archive.

## Inference

After preparing the data, dependencies, backbone, and a compatible report-generation checkpoint:

1. Modify the paths in `scripts/1-2.shallow_test_iuxray_conceptclip_l.sh` for your environment.
2. From the task directory, run:

```bash
cd scripts
bash 1-2.shallow_test_iuxray_conceptclip_l.sh
```
Predictions are saved to `$savepath/result/test_result.json` and references to `$savepath/result/test_refs.json`.

## Evaluation
Next, you can evaluate the inference results on the **IU-Xray** dataset using the evaluation tools provided in the `utils` directory.
> Remember to modify the paths in these files according to your environment settings.
1. Generating the results csv file.
```bash
cd downstream_evaluation/medical_report_generation/utils  # From the repository root
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

1. Prepare the data and base models above and modify the paths in `scripts/1-1.shallow_run_iuxray_conceptclip_l.sh` for your environment.
2. From the task directory, run:

```bash
mkdir -p scripts/logs
cd scripts
bash 1-1.shallow_run_iuxray_conceptclip_l.sh
```
Training logs are saved in `scripts/logs` and selected checkpoints under `$savepath/checkpoints`. Use a resulting checkpoint as `delta_file` for inference.
