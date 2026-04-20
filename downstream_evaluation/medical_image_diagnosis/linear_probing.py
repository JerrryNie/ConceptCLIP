import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
# sys.path.append('/home/ynieae/open_clip_med_pretrain/src/eval')
from utils import ConceptCLIPMultiLabelCLSEvalDataset, ConceptCLIPCLSEvalDataset
from utils import get_metrics, tensor_to_serializable, get_metrics_multilabel
from linear_probing._utils import get_features, shuffle_and_sample_data
from open_clip import create_model_and_transforms, get_tokenizer
from medmnist.evaluator import getACC, getAUC
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json

import open_clip
from transformers import AutoTokenizer, AutoModel, AutoConfig
from open_clip.transform import PreprocessCfg, image_transform_v2
import types

# # Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)
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

preprocess_cfg = {'interpolation': 'bicubic',
    'mean': [0.5, 0.5, 0.5],
    'resize_mode': 'squash',
    'size': 384,
    'std': [0.5, 0.5, 0.5]
}
pp_cfg = PreprocessCfg(**preprocess_cfg)
preprocess_train = image_transform_v2(pp_cfg, is_train=True)
preprocess_val = image_transform_v2(pp_cfg, is_train=False)
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

model.to(device)
model.eval()
model_name = model_name.replace('/', '_')
# Directory to save bootstrap results
bootstrap_result_dir = 'linear_probing_cache'

train_paths = ['data/meta/011_SIIM-ACR_CLS_CLIP_Train.json',]
test_paths = ['data/meta/011_SIIM-ACR_CLS_CLIP_Test.json']

for train_path, test_path in zip(train_paths, test_paths):
    batch_size = 64
    train_dataset = ConceptCLIPCLSEvalDataset(
        ann_path=train_path,
        data_path='data/images',
        transform=preprocess_train,
        tokenizer=tokenizer,
        device=device,
        context_length=77,
    )
    test_dataset = ConceptCLIPCLSEvalDataset(
        ann_path=test_path,
        data_path='data/images',
        transform=preprocess_val,
        tokenizer=tokenizer,
        device=device,
        context_length=77,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    # Calculate the image features
    train_features, train_labels = get_features(model, train_dataloader, ann_path=train_path, model_name=model_name, cache_dir='linear_probing_cache')
    test_features, test_labels = get_features(model, val_dataloader, ann_path=test_path, model_name=model_name, cache_dir='linear_probing_cache')

    # Define percentages for few-shot learning
    percentages = [1, 10, 100]

    # Dictionary to store metrics for each percentage
    all_metrics = {}

    for perc in percentages:
        print(f"Training with {perc}% of the data")
        task = f"linear-probe-{perc}%"
        anno_basename = os.path.basename(test_path)
        _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, anno_basename)
        if not os.path.exists(_cache_dir):
            os.makedirs(_cache_dir)
        # Shuffle and sample the data
        sampled_features, sampled_labels = shuffle_and_sample_data(train_features, train_labels, perc)

        # Perform logistic regression
        # classifier = OneVsRestClassifier(LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1))
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(sampled_features, sampled_labels)

        # Evaluate using the logistic regression classifier
        logits = classifier.predict_proba(test_features)

        # Calculate metrics and confidence intervals
        val_metrics, ci_metrics = get_metrics(y_true=torch.from_numpy(test_labels),
                                              y_score=torch.from_numpy(logits),
                                              n_resamples=1000)

        # Store metrics
        metrics = {}
        metrics.update(val_metrics)
        metrics.update({k: v for k, v in ci_metrics.items() if 'CI' in k})
        all_metrics[perc] = metrics

        # Save bootstrap metrics
        bootstrap_metrics = {k: v for k, v in ci_metrics.items() if 'Bootstrap' in k}
        print(f'bootstrap_metrics for {perc}% data: {bootstrap_metrics}')
        with open(os.path.join(_cache_dir, f'bootstrap_metrics.json'), 'w') as f:
            json.dump(tensor_to_serializable(bootstrap_metrics), f)
        print(f"Saved bootstrap metrics to {_cache_dir}")

        print(f"Metrics for {perc}% data: {metrics}")

    # Print all metrics
    print("All metrics:", all_metrics)