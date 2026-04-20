import sys
import torch
import os

import torch
from contextlib import suppress
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier

import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer, build_zero_shot_classifier
from utils import (ConceptCLIPMultiLabelCLSEvalDataset, evaluate_multilabel, evaluate, ConceptCLIPCLSEvalDataset, build_zero_shot_classifier,
                   build_zero_shot_concept_classifier, evaluate_with_local, evaluate_multilabel_with_local)
from transformers import AutoModel, AutoConfig
from open_clip.transform import PreprocessCfg, image_transform_v2
import argparse
import json
import types

# Initialize model and processor
model = AutoModel.from_pretrained('../../pre_training/src/pretrained_checkpoint/ConceptCLIP', trust_remote_code=True)
model_name = 'ConceptCLIP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
original_forward = model.forward


# 3. Define a new forward function with the desired parameters
def patched_forward(self, image=None, text=None, **kwargs):
    return original_forward(
        pixel_values=image,
        input_ids=text,
        **kwargs
    )

# 4. Bind the new forward method to the model instance
model.forward = types.MethodType(patched_forward, model)
class Args(argparse.Namespace):
  batch_size = 32
  device = torch.device('cuda')


args=Args()
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
preprocess_cfg = {'interpolation': 'bicubic',
 'mean': [0.5, 0.5, 0.5],
 'resize_mode': 'squash',
 'size': 384,
 'std': [0.5, 0.5, 0.5]}
pp_cfg = PreprocessCfg(**preprocess_cfg)
preprocess_val = image_transform_v2(pp_cfg, is_train=False)

model = model.eval()
model.to(device)

ann_path = 'data/meta/011_SIIM-ACR_CLS_CLIP_Test.json'
data_path = 'data/images'
dataset = ConceptCLIPCLSEvalDataset(
    ann_path=ann_path,
    data_path=data_path,
    transform=preprocess_val,
    tokenizer=tokenizer,
    device=device,
    context_length=77,
)
classnames = ['No Finding', 'Pneumothorax']
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=16,
    collate_fn=dataset.collate_fn,
    pin_memory=True,
    sampler=None,
    drop_last=False,
)

pathology_templates = [
    'a chest radiology presents {}',
]


templates = [lambda c: template.format(c) for template in pathology_templates]
global_classifier = build_zero_shot_classifier(model, tokenizer=tokenizer,
                                        classnames=classnames,
                                        templates=templates,
                                        device=device)

local_classifier = build_zero_shot_concept_classifier(model, tokenizer=tokenizer,
                                        classnames=classnames,
                                        templates=templates,
                                        device=device)

metrics, all_logits, all_labels = evaluate_with_local(model, dataloader, global_classifier, local_classifier, topk=128, alpha=0.5,
                                                      epoch=0, args=args, task='zero_shot', model_name=model_name, anno_path=ann_path,
                                                      n_resamples=1000, bootstrap_result_dir='zero_shot_cache')
print(metrics)