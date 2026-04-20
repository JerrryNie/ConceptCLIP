import sys
import torch
import argparse
import os

import torch
from PIL import Image, ImageFile
import logging
import torch.nn.functional as F
from contextlib import suppress
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import os
from tqdm import tqdm
from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union
from copy import deepcopy

# from medmnist.evaluator import getACC, getAUC
# from torcheval.metrics import MulticlassAUROC, BinaryAUROC
from torchmetrics import Metric
from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy, MulticlassAUROC, MultilabelAccuracy, MultilabelAUROC

import open_clip
from open_clip import get_tokenizer
from transformers import AutoModel, AutoConfig
from open_clip.transform import PreprocessCfg, image_transform_v2
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import roc_auc_score, accuracy_score

_ = torch.manual_seed(123)
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)
        # class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = model.text(texts)[0]
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def build_zero_shot_concept_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):

        class_embeddings = []
        for name in batch_classnames:
            class_embeding = []
            name_list = tokenizer.tokenizer.tokenize(name)
            for template in templates:
                template_name = template(name)
                template_name_list = tokenizer.tokenizer.tokenize(template_name)
                texts = tokenizer(template_name).cuda()
                indices_of_name = [template_name_list.index(value) + 1 for value in name_list if value in template_name_list]
                class_output = model(text=texts)
                class_embeding.append(class_output["text_token_features"][0][indices_of_name].mean(dim=0))
            class_embeddings.append(torch.stack(class_embeding, dim=0).mean(dim=0))
        class_embeddings = torch.stack(class_embeddings, dim=0)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


# Function
def evaluate_with_local(model, dataloader, global_classifier, local_classifier, topk, alpha,
                        epoch, args, task, model_name, anno_path, n_resamples=1000, 
                        bootstrap_result_dir='/data/ynieae/bootstrap_dir'):
    metrics = {}
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    num_samples = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                output = model(image=images)
            # break
            image_features = output["image_features"]
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = output["logit_scale"]
            global_core = (logit_scale * image_features @ global_classifier)
            global_core = torch.softmax(global_core, dim=-1)

            image_token_features = output["image_token_features"]
            concept_logit_scale = output["concept_logit_scale"]
            image_token_features = F.normalize(image_token_features, dim=-1)
            
            local_core = torch.topk(concept_logit_scale * image_token_features @ local_classifier, k=topk, dim=1)[0].mean(dim=1)
            local_core = torch.softmax(local_core, dim=-1)
            logits = alpha * local_core + (1 - alpha) * global_core
            logits = logits.cpu()
            labels = labels.cpu()

            all_logits.append(logits)
            all_labels.append(labels)

            batch_size = image_features.shape[0]
            num_samples += batch_size

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_metrics, ci_metrics = get_metrics(
        y_true=all_labels,
        y_score=all_logits,
        n_resamples=n_resamples,
    )
    metrics.update({
        **val_metrics,
        "epoch": epoch,
        "num_samples": num_samples
    })
    metrics.update({k:v for k, v in ci_metrics.items() if 'CI' in k})
    bootstrap_metrics = {k:v for k, v in ci_metrics.items() if 'Bootstrap' in k}

    # save bootstrap metrics
    anno_basename = os.path.basename(anno_path)
    _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, anno_basename)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    print(f'bootstrap_metrics: {bootstrap_metrics}')
    with open(os.path.join(_cache_dir, f'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {_cache_dir}")

    # Save evaluation metrics
    with open(os.path.join(_cache_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {_cache_dir}")
    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def evaluate(model, dataloader, epoch, args, classifier, task, model_name, anno_path, n_resamples=1000, 
             bootstrap_result_dir='/data/ynieae/bootstrap_dir'):
    metrics = {}
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    num_samples = 0

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(image=images)
            image_features = output["image_features"]
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = output["logit_scale"]
            logits = (logit_scale * image_features @ classifier)
            logits = torch.softmax(logits, dim=-1)
            logits = logits.cpu()
            labels = labels.cpu()
            all_logits.append(logits)
            all_labels.append(labels)

            batch_size = image_features.shape[0]
            num_samples += batch_size

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_metrics, ci_metrics = get_metrics(
        y_true=all_labels,
        y_score=all_logits,
        n_resamples=n_resamples,
    )
    metrics.update({
        **val_metrics,
        "epoch": epoch,
        "num_samples": num_samples
    })
    metrics.update({k:v for k, v in ci_metrics.items() if 'CI' in k})
    bootstrap_metrics = {k:v for k, v in ci_metrics.items() if 'Bootstrap' in k}

    # save bootstrap metrics
    anno_basename = os.path.basename(anno_path)
    _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, anno_basename)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    print(f'bootstrap_metrics: {bootstrap_metrics}')
    with open(os.path.join(_cache_dir, f'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {_cache_dir}")

    # Save evaluation metrics
    with open(os.path.join(_cache_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {_cache_dir}")

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def get_metrics(y_true, y_score, n_resamples=1000):
    metrics = {}
    y_true_np = y_true.numpy()
    y_score_np = y_score.numpy()

    metrics['ACC'] = getACC(y_true_np, y_score_np, task="multi-class")
    metrics['AUC'] = getAUC(y_true_np, y_score_np, task="multi-class")
    AUROC_metric = MulticlassAUROC(num_classes=y_score_np.shape[1])
    metrics['torch_AUROC'] = AUROC_metric(y_score, y_true).item()

    # Calculate confidence intervals
    ci_metrics = calculate_confidence_intervals(y_true, y_score, n_resamples)

    return metrics, ci_metrics


def calculate_confidence_intervals(y_true, y_score, n_resamples=1000):
    from torchmetrics.classification import Accuracy, MulticlassAUROC
    from torchmetrics.wrappers import BootStrapper

    # Initialize BootStrappers with the desired metrics
    # y_true = y_true.to(torch.float32)
    y_score = y_score.to(torch.float32)
    num_classes = y_score.shape[1]
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float32)

    bootstrap_acc = BootStrapper(
        Accuracy(task='multiclass', num_classes=num_classes),
        num_bootstraps=n_resamples,
        quantile=quantiles,
        raw=True
    )

    bootstrap_auroc = BootStrapper(
        MulticlassAUROC(num_classes=num_classes),
        num_bootstraps=n_resamples,
        quantile=quantiles,
        raw=True
    )

    # Convert probabilities to class predictions for accuracy
    y_pred = torch.argmax(y_score, dim=1)

    # Batch update the BootStrappers
    bootstrap_acc.update(y_pred, y_true)
    bootstrap_auroc.update(y_score, y_true)

    # Compute the confidence intervals
    acc_result = bootstrap_acc.compute()
    auroc_result = bootstrap_auroc.compute()

    ci_metrics = {
        'ACC_CI': [
            acc_result['quantile'][0].item(),
            acc_result['quantile'][1].item()
        ],
        'AUROC_CI': [
            auroc_result['quantile'][0].item(),
            auroc_result['quantile'][1].item()
        ],
        'Bootstrap_ACC': acc_result,
        'Bootstrap_AUROC': auroc_result
    }

    return ci_metrics


def getACC(y_true, y_score, task):
    y_pred = np.argmax(y_score, axis=1)
    acc = np.mean(y_pred == y_true)
    return acc


def getAUC(y_true, y_score, task):
    from sklearn.metrics import roc_auc_score
    if task == "multi-class":
        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except ValueError:
            auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_score)
    return auc


class BiomedCLIPCLSEvalDataset(Dataset):
    def __init__(self,
                 ann_path,
                 data_path,
                 transform,
                 tokenizer,
                 device,
                 context_length=256,
                 ):
        self.context_length = context_length
        self.img_dir = data_path
        self.device = device
        self.ann_path = ann_path

        self.transform = transform
        self.tokenizer = tokenizer
        self.prompt = ''
        # load annoations
        self._load_annotation()

    def _load_annotation(self):
        # with open(self.ann_path, 'r') as f:
        #     self.annts = [json.loads(line) for line in f.read().splitlines()]
        # self.texts = [ann['text_list'][0] for ann in self.annts]
        # self.img_paths = [ann['image_info'][0]['image_name'] for ann in self.annts]
        with open(self.ann_path, 'r') as f:
            data = json.load(f)
        # self.labels = data['label_set']
        self.labels = [item for item in data['label_set'] if item != '']
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for idx, label in enumerate(self.labels)}
        self.annts = data['annotations']

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        info = self.annts[index]
        # text = info['text_list'][0]
        label = self.label2idx[info['label_text']]
        info['label_text'] != ''
        if isinstance(info['image'], list):
            info['image'] = info['image'][0]
        img_path = os.path.join(self.img_dir, info['image'])
        assert os.path.exists(img_path), f"Image not found: {img_path}"
        image = Image.open(img_path).convert('RGB')

        return image, label
    
    def collate_fn(self, batch):
        # print(batch)
        images = [image for image, _ in batch]
        labels = [label for _, label in batch]
        images = torch.stack([self.transform(image) for image in images])
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return images, labels

def evaluate_multilabel(model, dataloader, epoch, args, classifier, task, model_name, anno_path, n_resamples=1000, 
             bootstrap_result_dir='/data/ynieae/bootstrap_dir'):
    metrics = {}
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    num_samples = 0

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # with autocast():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(image=images)
            image_features = output["image_features"]
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = output["logit_scale"]
            logits = (logit_scale * image_features @ classifier)
            logits = torch.sigmoid(logits)
            logits = logits.cpu()
            labels = labels.cpu()
            all_logits.append(logits)
            all_labels.append(labels)

            batch_size = image_features.shape[0]
            num_samples += batch_size

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_metrics, ci_metrics = get_metrics_multilabel(
        y_true=all_labels,
        y_score=all_logits,
        n_resamples=n_resamples,
    )
    metrics.update({
        **val_metrics,
        "epoch": epoch,
        "num_samples": num_samples
    })
    metrics.update({k:v for k, v in ci_metrics.items() if 'CI' in k})
    bootstrap_metrics = {k:v for k, v in ci_metrics.items() if 'Bootstrap' in k}

    # save bootstrap metrics
    anno_basename = os.path.basename(anno_path)
    _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, anno_basename)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    print(f'bootstrap_metrics: {bootstrap_metrics}')
    with open(os.path.join(_cache_dir, f'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {_cache_dir}")

    # Save evaluation metrics
    with open(os.path.join(_cache_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {_cache_dir}")

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def evaluate_multilabel_with_local(model, dataloader, global_classifier, local_classifier, topk, alpha,
                                   epoch, args, task, model_name, anno_path, n_resamples=1000, 
                                   bootstrap_result_dir='/data/ynieae/bootstrap_dir'):
    metrics = {}
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    num_samples = 0

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # with autocast():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(image=images)
            image_features = output["image_features"]
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = output["logit_scale"]
            global_core = (logit_scale * image_features @ global_classifier)
            global_core = torch.sigmoid(global_core)

            image_token_features = output["image_token_features"]
            concept_logit_scale = output["concept_logit_scale"]
            image_token_features = F.normalize(image_token_features, dim=-1)
            
            local_core = torch.topk(concept_logit_scale * image_token_features @ local_classifier, k=topk, dim=1)[0].mean(dim=1)
            local_core = torch.sigmoid(local_core)
            logits = alpha * local_core + (1 - alpha) * global_core
            logits = logits.cpu()
            labels = labels.cpu()

            all_logits.append(logits)
            all_labels.append(labels)

            batch_size = image_features.shape[0]
            num_samples += batch_size

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_metrics, ci_metrics = get_metrics_multilabel(
        y_true=all_labels,
        y_score=all_logits,
        n_resamples=n_resamples,
    )
    metrics.update({
        **val_metrics,
        "epoch": epoch,
        "num_samples": num_samples
    })
    metrics.update({k:v for k, v in ci_metrics.items() if 'CI' in k})
    bootstrap_metrics = {k:v for k, v in ci_metrics.items() if 'Bootstrap' in k}

    # save bootstrap metrics
    anno_basename = os.path.basename(anno_path)
    _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, anno_basename)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    print(f'bootstrap_metrics: {bootstrap_metrics}')
    with open(os.path.join(_cache_dir, f'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {_cache_dir}")

    # Save evaluation metrics
    with open(os.path.join(_cache_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {_cache_dir}")
    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def get_metrics_multilabel(y_true, y_score, n_resamples=1000):
    metrics = {}
    y_true_np = y_true.numpy()
    y_score_np = y_score.numpy()

    metrics['AUC'] = getAUCMultiLabel(y_true_np, y_score_np, task="multi-label")

    AUROC_metric = MultilabelAUROC(num_labels=y_score_np.shape[1])
    metrics['torch_AUROC'] = AUROC_metric(y_score, y_true).item()

    ci_metrics = calculate_confidence_intervals_multilabel(y_true, y_score, n_resamples)

    return metrics, ci_metrics


def calculate_confidence_intervals_multilabel(y_true, y_score, n_resamples=1000):
    # Initialize BootStrappers with the desired metrics
    # y_true = y_true.to(torch.float32)
    y_score = y_score.to(torch.float32)
    num_labels = y_score.shape[1]
    quantiles = torch.tensor([0.025, 0.975])

    bootstrap_acc = BootStrapper(
        MultilabelAccuracy(num_labels=num_labels, average='macro'),
        num_bootstraps=n_resamples,
        quantile=quantiles,
        raw=True
    )

    bootstrap_auroc = BootStrapper(
        MultilabelAUROC(num_labels=num_labels, average='macro'),
        num_bootstraps=n_resamples,
        quantile=quantiles,
        raw=True
    )

    # Threshold probabilities to get predictions
    y_pred = (y_score >= 0.5).float()

    # Batch update the BootStrappers
    # print()
    bootstrap_acc.update(y_pred, y_true)
    bootstrap_auroc.update(y_score, y_true)

    # Compute the confidence intervals
    acc_result = bootstrap_acc.compute()
    auroc_result = bootstrap_auroc.compute()

    ci_metrics = {
        'ACC_CI': [
            acc_result['quantile'][0].item(),
            acc_result['quantile'][1].item()
        ],
        'AUROC_CI': [
            auroc_result['quantile'][0].item(),
            auroc_result['quantile'][1].item()
        ],
        'Bootstrap_ACC': acc_result,
        'Bootstrap_AUROC': auroc_result
    }

    return ci_metrics

def getAUCMultiLabel(y_true, y_score, task):
    from sklearn.metrics import roc_auc_score
    if task == "multi-label":
        auc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
    else:
        # Handle other tasks if necessary
        auc = roc_auc_score(y_true, y_score)
    return auc


class BiomedCLIPMultiLabelCLSEvalDataset(Dataset):
    def __init__(self,
                 ann_path,
                 data_path,
                 transform,
                 tokenizer,
                 device,
                 context_length=256,
                 labels=None,
                 ):
        self.context_length = context_length
        self.img_dir = data_path
        self.device = device
        self.ann_path = ann_path

        self.transform = transform
        self.tokenizer = tokenizer
        self.prompt = ''
        # load annoations
        self._load_annotation(labels=labels)

    def _load_annotation(self, labels=None):
        # with open(self.ann_path, 'r') as f:
        #     self.annts = [json.loads(line) for line in f.read().splitlines()]
        # self.texts = [ann['text_list'][0] for ann in self.annts]
        # self.img_paths = [ann['image_info'][0]['image_name'] for ann in self.annts]
        with open(self.ann_path, 'r') as f:
            data = json.load(f)
        # self.labels = data['label_set']
        if labels is not None:
            assert isinstance(labels, list)
            self.labels = labels
        else:
            self.labels = [item for item in data['label_set'] if item != '']
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for idx, label in enumerate(self.labels)}
        self.annts = data['annotations']

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        info = self.annts[index]
        # text = info['text_list'][0]
        # label = self.label2idx[info['label_text']]
        # print(f'len(self.labels): {len(self.labels)}')
        label = [0] * len(self.labels)
        for item in info['label']:
            label[self.label2idx[item]] = 1
        if isinstance(info['image'], list):
            info['image'] = info['image'][0]
        if info['image'].startswith('/'):
            info['image'] = info['image'][1:]
        img_path = os.path.join(self.img_dir, info['image'])
        assert os.path.exists(img_path), f"Image not found: {img_path}"
        image = Image.open(img_path).convert('RGB')

        return image, label
    
    def collate_fn(self, batch):
        # print(batch)
        images = [image for image, _ in batch]
        labels = [label for _, label in batch]
        images = torch.stack([self.transform(image) for image in images])
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return images, labels