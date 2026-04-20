import torch
import sys
import os
import json
import logging
from tqdm import tqdm
import torch.nn.functional as F
# sys.path.append('../src/eval')
import math
from utils import get_metrics, get_metrics_multilabel, tensor_to_serializable

def evaluate(model, dataloader, classifier, device, epoch=0, n_resamples=1000, 
             bootstrap_result_dir='fully_finetuning_cache'):
    metrics = {}
    model.to(device)
    model.eval()

    num_samples = 0

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            features = model.encode_image(images)
            if isinstance(features, tuple):
                features = features[0]
            if isinstance(features, dict):
                features = features['image_features']
            features = F.normalize(features, dim=-1)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            probs = probs.cpu()
            labels = labels.cpu()
            all_logits.append(probs)
            all_labels.append(labels)

            num_samples += images.size(0)

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
    metrics.update({k: v for k, v in ci_metrics.items() if 'CI' in k})
    bootstrap_metrics = {k: v for k, v in ci_metrics.items() if 'Bootstrap' in k}

    # Save bootstrap metrics
    if not os.path.exists(bootstrap_result_dir):
        os.makedirs(bootstrap_result_dir)
    with open(os.path.join(bootstrap_result_dir, 'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {bootstrap_result_dir}")

    # Save evaluation metrics
    with open(os.path.join(bootstrap_result_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {bootstrap_result_dir}")

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def evaluate_multilabel(model, dataloader, classifier, device, epoch=0, n_resamples=1000, 
                        bootstrap_result_dir='fully_finetuning_cache'):
    metrics = {}
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
            
            features = model.encode_image(images)
            if isinstance(features, tuple):
                features = features[0]
            if isinstance(features, dict):
                features = features['image_features']
            features = F.normalize(features, dim=-1)
            logits = classifier(features)
            probs = torch.sigmoid(logits)
            probs = probs.cpu()
            labels = labels.cpu()
            all_logits.append(probs)
            all_labels.append(labels)

            num_samples += images.size(0)

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

    # Save bootstrap metrics
    if not os.path.exists(bootstrap_result_dir):
        os.makedirs(bootstrap_result_dir)
    with open(os.path.join(bootstrap_result_dir, 'bootstrap_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(bootstrap_metrics), f)
    print(f"Saved bootstrap metrics to {bootstrap_result_dir}")

    # Save evaluation metrics
    with open(os.path.join(bootstrap_result_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(tensor_to_serializable(metrics), f)
    print(f"Saved evaluation metrics to {bootstrap_result_dir}")

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v}" for k, v in metrics.items()])
    )

    return metrics, all_logits, all_labels


def build_optimizer(model, layer_decay, lr, fix_layer=-1):

    params_name = None

    params, param_group_names = param_groups_lrd(model, fix_layer, layer_decay=layer_decay)
    params_name = []
    for k, v in param_group_names.items():
        params_name += v["params"]
    optimizer = torch.optim.AdamW(params, lr=lr)

    for name, param in model.named_parameters():
        if name not in params_name:
            param.requires_grad = False

    return optimizer

def param_groups_lrd(model, fix_layer=-1, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}  

    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    elif hasattr(model, "transformer"):
        num_layers = model.transformer.layers + 1
    elif hasattr(model, "visual") and hasattr(model.visual, "layer1"):
        num_layers = sum(len(layer) for layer in [
            model.visual.layer1,
            model.visual.layer2,
            model.visual.layer3,
            model.visual.layer4
        ]) + 1
    elif hasattr(model, 'vision_model'):
        num_layers = len(model.vision_model.encoder.layers) + 1
    elif hasattr(model, 'image_encoder') and hasattr(model.image_encoder, '_blocks'):
        num_layers = len(model.image_encoder._blocks) + 1
    elif hasattr(model, 'visual') and hasattr(model.visual, "transformer") and hasattr(model.visual.transformer, "resblocks"):
        num_layers = len(model.visual.transformer.resblocks) + 1
    else:
        num_layers = len(model.visual.trunk.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        # import pdb; pdb.set_trace()
        if hasattr(model, 'visual') and hasattr(model.visual, "trunk") and hasattr(model.visual.trunk, "blocks"):
            layer_id = get_layer_id_for_vit(n, num_layers)
        else:
            layer_id = get_layer_id_for_clip(n, num_layers)

        if layer_id > fix_layer:

            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)
    # print(param_group_names)
    for k, v in param_group_names.items():
        print("############################################")
        print(f"Group {k}")
        print(f"lr_scale: {v['lr_scale']}")
        print(f"weight_decay: {v['weight_decay']}")
        print(f"params: {v['params']}")
    return list(param_groups.values()), param_group_names


def get_layer_id_for_vit(name, num_layers):
    name = name.replace("visual.trunk.", "")
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

def get_layer_id_for_clip(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """

    if name in ['cls_token', 'pos_embed', "class_embedding"]: 
        return 0
    elif name.startswith('patch_embed'):  
        return 0
    elif name.startswith('conv1'):  
        return 0
    elif name.startswith('ln_pre'):  
        return 0
    elif name.startswith('positional_embedding'):  
        return 0
    elif name.startswith('transformer.resblocks'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers

def adjust_learning_rate(optimizer, epoch, warmup_epochs=1, epochs=10, lr=1e-4, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr