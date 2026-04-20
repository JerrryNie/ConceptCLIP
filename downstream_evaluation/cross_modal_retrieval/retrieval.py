import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.append('../../pre_training/src/open_clip_train')
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from datasets.pmc_dataset import PmcDataset
import sys
from open_clip_train.data import get_data
from open_clip import create_model_and_transforms
import torch
from PIL import Image
import logging
import torch.nn.functional as F
from contextlib import suppress
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import os
from tqdm import tqdm
from model_loader import MODELS
import os
import json
from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj

def get_pmc_data(args, preprocess_fn, epoch=0, tokenizer=None):

    input_filename = args.val_data
    dataset_cls = PmcDataset
    dataset = dataset_cls(
        args,
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        is_train=False
    )

    num_samples = len(dataset)
    sampler = None
    shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader


def evaluate(model, dataloader, epoch, args, tb_writer=None,
             n_resamples=1000, model_name=None,
             task='retrieval',
             bootstrap_result_dir='retrieval_cache'):
    metrics = {}
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    num_samples = 0

    cumulative_total_loss = 0.0
    cumulative_match_loss = 0.0
    cumulative_mlm_loss = 0.0
    cumulative_mim_loss = 0.0

    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, texts = batch[:2]
            image_features = model.encode_image(images.to(args.device))
            if isinstance(image_features, tuple):
                image_features = image_features[0]
            if isinstance(image_features, dict):
                image_features = image_features["image_features"]
            text_features = model.encode_text(texts.to(args.device))
            if isinstance(text_features, tuple):
                text_features = text_features[0]
            if isinstance(text_features, dict):
                text_features = text_features["text_features"]
            logit_scale = model.logit_scale.exp()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logits_per_image = 100.0 * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            match_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
            mlm_loss = match_loss

            total_loss = match_loss

            cumulative_total_loss += total_loss * batch_size
            num_samples += batch_size

        val_metrics = get_metrics_gpu(
            image_features=torch.cat(all_image_features).float(),
            text_features=torch.cat(all_text_features).float(),
            logit_scale=logit_scale.float(),
        )
        metrics.update({
            **val_metrics,
            "val_loss": (cumulative_total_loss / num_samples).item(),
            "epoch": epoch,
            "num_samples": num_samples
        })
        # Calculate bootstrap metrics
        ci_metrics = bootstrap_retrieval(torch.cat(all_image_features), torch.cat(all_text_features), logit_scale, n_resamples)
        metrics.update({k: v for k, v in ci_metrics.items() if 'CI' in k})
        bootstrap_metrics = {k: v for k, v in ci_metrics.items() if 'Bootstrap' in k}

        anno_path = args.val_data
        _cache_dir = os.path.join(bootstrap_result_dir, task, model_name, os.path.basename(anno_path))
        if not os.path.exists(_cache_dir):
            os.makedirs(_cache_dir)
        with open(os.path.join(_cache_dir, 'bootstrap_metrics.json'), 'w') as f:
            json.dump(tensor_to_serializable(bootstrap_metrics), f)
        print(f"Saved bootstrap metrics to {_cache_dir}")
        with open(os.path.join(_cache_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(tensor_to_serializable(metrics), f)
        print(f"Saved evaluation metrics to {_cache_dir}")

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics


def get_metrics_gpu(image_features, text_features, logit_scale):
    metrics = {}
    batch_size = 128
    num_batches = (len(image_features) + batch_size - 1) // batch_size
    logits_per_image = []
    for i in tqdm(range(num_batches), desc='Computing logits'):
        logits_per_image.append(
            (logit_scale * image_features[i * batch_size:(i + 1) * batch_size] @ text_features.t()).detach().cpu()
        )
    logits_per_image = torch.cat(logits_per_image)
    logits_per_text = logits_per_image.t()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1).cuda()

    for name, logit in tqdm(logits.items(), desc='Computing metrics'):
        preds = []
        for i in tqdm(range(num_batches), desc='Computing ranking'):
            part_ranking = torch.argsort(logit[i * batch_size:(i + 1) * batch_size].cuda(), descending=True)
            part_preds = torch.where(part_ranking == ground_truth[i * batch_size:(i + 1) * batch_size])[1]
            preds.append(part_preds)
        preds = torch.cat(preds)
        mean_rank = preds.float().mean() + 1
        median_rank = torch.median(preds.float()).floor() + 1
        metrics[f"{name}_mean_rank"] = mean_rank.item()
        metrics[f"{name}_median_rank"] = median_rank.item()

        for k in [1, 5, 10, 50, 200]:
            print(f"Computing R@{k}")
            metrics[f"{name}_R@{k}"] = (preds < k).float().mean().item()

    return metrics

def bootstrap_retrieval(image_features, text_features, logit_scale, n_resamples=1):
    device = image_features.device
    with torch.no_grad():
        logits_per_image = (logit_scale * image_features @ text_features.t())  
        logits_per_text = logits_per_image.t()  
    ground_truth = torch.arange(logits_per_image.shape[0], device=device)

    def compute_ranks(logits):
        # Returns the rank positions of ground_truth for each row
        sorted_indices = torch.argsort(logits, dim=1, descending=True)
        # Positions of each row's ground_truth in sorted_indices
        return (sorted_indices == ground_truth.view(-1,1)).nonzero()[:, 1]

    ranks_i2t = compute_ranks(logits_per_image)
    ranks_t2i = compute_ranks(logits_per_text)

    def compute_r_at_k(ranks, k):
        return (ranks < k).float().mean()

    # Vectorized metrics on full set
    base_metrics = {}
    for name, ranks in [("i2t", ranks_i2t), ("t2i", ranks_t2i)]:
        for k in [1, 5, 10, 50, 200]:
            base_metrics[f"{name}_R@{k}"] = compute_r_at_k(ranks, k).item()

    # Bootstrapping fully on GPU
    n_samples = logits_per_image.shape[0]
    bootstrap_scores = {f"Bootstrap_{k}": [] for k in base_metrics.keys()}
    for _ in tqdm(range(n_resamples)):
        indices = torch.randint(n_samples, (n_samples,), device=device)
        # For i2t
        ranks_i2t_sampled = compute_ranks(logits_per_image[indices][:, indices])
        # For t2i
        ranks_t2i_sampled = compute_ranks(logits_per_text[indices][:, indices])
        for k in [1, 5, 10, 50, 200]:
            bootstrap_scores[f"Bootstrap_i2t_R@{k}"].append(compute_r_at_k(ranks_i2t_sampled, k).item())
            bootstrap_scores[f"Bootstrap_t2i_R@{k}"].append(compute_r_at_k(ranks_t2i_sampled, k).item())

    # CI
    ci_metrics = {}
    for key, vals in bootstrap_scores.items():
        arr = np.array(vals)
        lower, upper = np.percentile(arr, [2.5, 97.5])
        ci_metrics[f"{key.replace('Bootstrap_', '')}_CI"] = [float(lower), float(upper)]

    final_res = {**base_metrics, **ci_metrics, **bootstrap_scores}
    return final_res

def load_model_and_eval_wds(args, model_key, n_resamples=1):
    model_name, model, preprocess_train, preprocess_val, tokenizer, out_dim = MODELS[model_key]()
    print(preprocess_val)
    dataloader = get_pmc_data(args, preprocess_val, tokenizer=tokenizer)
    model.to(args.device)
    model.eval()
    print(evaluate(model, dataloader=dataloader, epoch=10, args=args, tb_writer=None, n_resamples=n_resamples, model_name=model_name))


class Args(argparse.Namespace):
  batch_size = 256
  workers = 32
  val_data = 'data/meta/quilt1m_test.jsonl'
  image_dir = 'data/images'
  csv_img_key = 'image'
  csv_caption_key = 'caption'
  meta_key = None
  csv_separator = '\t'
  device = torch.device('cuda')

args=Args()

load_model_and_eval_wds(args, model_key='conceptclip')