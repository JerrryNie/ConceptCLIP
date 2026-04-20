import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def define_loss(args):
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return criterion
