import sys
sys.path.append('/home/ynieae/open_clip_med_pretrain/src')
from open_clip_train.datasets import PmcDataset
from open_clip import get_tokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import os

from transformers import AutoTokenizer

from open_clip.transform import PreprocessCfg, image_transform_v2

class Args(argparse.Namespace):
    csv_img_key = 'image'
    csv_caption_key = 'caption'
    meta_key = 'umls_meta_info'
    csv_separator = ','
    image_dir = '/project/medimgfmod/Generalist/113_PMC_0924_336x336_untar'
    aug_cfg = {'scale': (0.4, 1.0), 'color_jitter': (0.32, 0.32, 0.32, 0.08), 'color_jitter_prob': 0.8, 'gray_scale_prob': 0.2}
    batch_size = 1024
    workers = 16
  

args=Args()

pp_cfg = PreprocessCfg()
print(f'pp_cfg: {pp_cfg}')

preprocess_val = image_transform_v2(
    pp_cfg,
    is_train=False,
    aug_cfg=args.aug_cfg,
)

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
tokenizer.context_length = 77
dataset = PmcDataset(
    args,
    input_filename='/project/medimgfmod/Generalist/113_PMC_0924_336x336_untar_min.jsonl',
    transforms=preprocess_val,
    tokenizer=tokenizer,
    is_train=False
)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
    drop_last=False,
)
for batch in tqdm(dataloader):
    pass
