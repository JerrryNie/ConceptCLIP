import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
import logging
import argparse

from sklearn.preprocessing import label_binarize

# Load the model and tokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
from open_clip.transform import PreprocessCfg, image_transform_v2

# Import evaluation functions
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
from torchmetrics.wrappers import BootStrapper

import sys
import time
# breakpoint()
# sys.path.append('../eval')
from utils import BiomedCLIPCLSEvalDataset, BiomedCLIPMultiLabelCLSEvalDataset
from _utils import evaluate, evaluate_multilabel, build_optimizer, adjust_learning_rate
from config import META
from model_loader import MODELS
import random
from torch.utils.data import SubsetRandomSampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning script')
    parser.add_argument('--arch', type=str, default='conceptclip', help='Name of the dataset')
    parser.add_argument('--modality', type=str, default='', help='Modality of the dataset')
    parser.add_argument('--dataset-name', type=str, default='SIIM-ACR', help='Name of the dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Number of accumulation steps')
    parser.add_argument('--task', type=str, default='fully_finetuning', help='Task name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cache-dir', type=str, default='fully_finetuning_cache', help='Directory for cache')
    parser.add_argument('--few-shot', type=float, default=100.0, help='Percentage of the training set to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the cache directory')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert args.arch in MODELS, f"Model {args.arch} not found in MODELS"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loader = MODELS[args.arch] if args.arch != 'modality_specific' else MODELS[args.arch][args.modality]
    model_name, model, preprocess_train, preprocess_val, tokenizer, output_dim = model_loader()
    model.to(device)

    dataset_name = args.dataset_name
    train_path = META[dataset_name]['train_path']
    test_path = META[dataset_name]['test_path']
    cls_type = META[dataset_name]['cls_type']
    evaluate_fn = evaluate if cls_type == 'mcls' else evaluate_multilabel
    dataset_cls = BiomedCLIPCLSEvalDataset if cls_type == 'mcls' else BiomedCLIPMultiLabelCLSEvalDataset
    if 'data_path' in META[dataset_name]:
        train_image_dir = test_image_dir = META[dataset_name]['data_path']
    else:
        train_image_dir = META[dataset_name]['train_data_path']
        test_image_dir = META[dataset_name]['test_data_path']
    if 'epochs' in META[dataset_name]:
        args.epochs = META[dataset_name]['epochs']
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    with open(train_path, 'r') as f:
        train_meta = json.load(f)

    cache_dir = args.cache_dir
    anno_basename = os.path.basename(test_path)
    if args.few_shot < 100.0:
        _cache_dir = os.path.join(cache_dir, args.task, model_name, anno_basename, f"few_shot_{args.few_shot}")
    else:
        _cache_dir = os.path.join(cache_dir, args.task, model_name, anno_basename)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    if os.path.exists(os.path.join(_cache_dir, 'evaluation_metrics.json')) and not args.overwrite:
        print(f'_cache_dir already exists: {_cache_dir}')
        print(f"Model {model_name} has been evaluated on {anno_basename}")
        exit(0)

    training_config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "task": args.task
    }

    with open(os.path.join(_cache_dir, 'training_config.json'), 'w') as config_file:
        json.dump(training_config, config_file, indent=2)

    train_dataset = dataset_cls(
        ann_path=train_path,
        data_path=train_image_dir,
        transform=preprocess_train,
        tokenizer=tokenizer,
        device=device,
        context_length=77,
    )
    test_dataset = dataset_cls(
        ann_path=test_path,
        data_path=test_image_dir,
        transform=preprocess_val,
        tokenizer=tokenizer,
        device=device,
        context_length=77,
    )

    few_shot_fraction = args.few_shot / 100.0
    train_size = len(train_dataset)
    indices = list(range(train_size))
    random.shuffle(indices)
    few_shot_size = int(train_size * few_shot_fraction)
    subset_indices = indices[:few_shot_size]
    train_sampler = SubsetRandomSampler(subset_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=128,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=64,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    # Modify the model for classification
    num_classes = len(train_dataset.labels)
    # breakpoint()
    model.classifier = nn.Linear(output_dim, num_classes).to(device)

    for n, p in model.named_parameters():
        if "text" in n:
            p.requires_grad = False
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    learning_rate = args.lr
    if 'linear_probes' not in args.task:
        optimizer = build_optimizer(model=model, layer_decay=0.75, lr=learning_rate)
    else:
        print("Using AdamW optimizer")
        optimizer = AdamW(model.classifier.parameters(), lr=learning_rate)
    # Define loss function and optimizer
    if cls_type == 'mcls':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Fine-tuning loop without validation
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        print(f"Epoch {epoch+1}")
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch") as t:
            # for i, (images, labels) in enumerate(train_dataloader):
            for i, (images, labels) in enumerate(t):
                if 'linear_probes' not in args.task:
                    adjust_learning_rate(optimizer, i / len(train_dataloader) + epoch, warmup_epochs=2, epochs=num_epochs, lr=learning_rate, min_lr=1e-6)
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                with autocast():
                    features = model.encode_image(images)
                    if isinstance(features, tuple):
                        features = features[0]
                    if isinstance(features, dict):
                        features = features['image_features']
                    outputs = model.classifier(features)
                    if cls_type != 'mcls':
                        labels = labels.float()
                    else:
                        labels = labels.long()
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                # Backward and optimize
                loss = loss / accumulation_steps
                # Use scaler for backprop
                scaler.scale(loss).backward()
                # loss.backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                    optimizer.zero_grad()
                    
                t.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # scheduler.step()
        # print(f"Learning rate: {scheduler.get_last_lr()}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Conduct evaluation
        classifier = model.classifier

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(_cache_dir, 'fine_tuned_model.pth'))
    print(f"Model saved to {os.path.join(_cache_dir, 'fine_tuned_model.pth')}")

    # Conduct evaluation
    classifier = model.classifier

    metrics, all_logits, all_labels = evaluate_fn(
        model=model,
        dataloader=val_dataloader,
        classifier=classifier,
        device=device,
        epoch=num_epochs,
        n_resamples=1000 if 'CheXpert'.lower() not in dataset_name.lower() else 50,
        bootstrap_result_dir=_cache_dir
    )

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")