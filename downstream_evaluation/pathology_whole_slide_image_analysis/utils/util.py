import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import Sampler


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter:
    def __init__(self, fold):
        self.fold = fold
        self.header = ["folds", "best_epoch", "Macro_AUC", "Macro_ACC", "Macro_F1", "Weighted_AUC", "Weighted_ACC", "Weighted_F1"]
        self.rows = []

    def updata(self, epoch, val_score, test_score=None):
        # convert the tensor to float
        val_score = {k: v.item() for k, v in val_score.items()}
        if test_score is not None:
            test_score = {k: v.item() for k, v in test_score.items()}
        row = [len(self.rows)]
        row.append(epoch)
        row.append(round(val_score["Macro_AUC"], 4))
        row.append(round(val_score["Macro_ACC"], 4))
        row.append(round(val_score["Macro_F1"], 4))
        row.append(round(val_score["Weighted_AUC"], 4))
        row.append(round(val_score["Weighted_ACC"], 4))
        row.append(round(val_score["Weighted_F1"], 4))
        self.rows.append(row)
        if test_score is not None:
            row = [len(self.rows)]
            row.append(epoch)
            row.append(str(round(test_score["Macro_AUC_mean"], 4)) + "±" + str(round(test_score["Macro_AUC_std"], 4)))
            row.append(str(round(test_score["Macro_ACC_mean"], 4)) + "±" + str(round(test_score["Macro_ACC_std"], 4)))
            row.append(str(round(test_score["Macro_F1_mean"], 4)) + "±" + str(round(test_score["Macro_F1_std"], 4)))
            row.append(str(round(test_score["Weighted_AUC_mean"], 4)) + "±" + str(round(test_score["Weighted_AUC_std"], 4)))
            row.append(str(round(test_score["Weighted_ACC_mean"], 4)) + "±" + str(round(test_score["Weighted_ACC_std"], 4)))
            row.append(str(round(test_score["Weighted_F1_mean"], 4)) + "±" + str(round(test_score["Weighted_F1_std"], 4)))
            self.rows.append(row)

    def save(self, path):
        print("save evaluation resluts to", path)
        if self.fold > 1:
            means = ["mean", "mean"]
            stds = ["std", "std"]
            for i in range(2, 8):
                means.append(round(np.mean([r[i] for r in self.rows]), 4))
                stds.append(round(np.std([r[i] for r in self.rows]), 4))
            self.rows.append(means)
            self.rows.append(stds)
        with open(path, "a", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerows(self.rows)

