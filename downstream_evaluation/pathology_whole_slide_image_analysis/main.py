import os
import csv
import time

from datasets.Subtyping import Dataset_Subtyping
from utils.options import parse_args
from utils.util import set_seed, CV_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        results_dir = "./results/results_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print("[log dir] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    dataset = Dataset_Subtyping(root=args.root, csv_file=args.csv_file, feature=args.feature)
    # training and evaluation
    meter = CV_Meter(dataset.num_folds)
    args.num_classes = dataset.num_classes
    args.n_features = dataset.n_features
    args.num_folds = dataset.num_folds
    for fold in range(dataset.num_folds):
        splits = dataset.get_fold(fold)
        loaders = [DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(split)) for split in splits]
        # build model, criterion, optimizer, schedular
        #################################################
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            from models.ABMIL.engine import Engine

            model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)
        # start training
        if not args.evaluate:
            if args.num_folds > 1:
                val_scores, best_epoch = engine.learning(model, loaders, criterion, optimizer, scheduler)
                meter.updata(best_epoch, val_scores)
            else:
                val_scores, test_scores, best_epoch = engine.learning(model, loaders, criterion, optimizer, scheduler)
                meter.updata(best_epoch, val_scores, test_scores)
    if not args.evaluate:
        meter.save(os.path.join(results_dir, "result.csv"))


if __name__ == "__main__":

    args = parse_args()
    results = main(args)
    print("finished!")
