import os
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from typing import Dict
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.bootstrapping import BootStrapper
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC

import torch
import torch.nn.functional as F


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.num_folds > 1:
            self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        else:
            self.results_dir = results_dir
        # tensorboard
        # if args.log_data:
        #     self.writer = SummaryWriter(self.results_dir, flush_secs=15)
        self.val_scores = None
        self.test_scores = None if self.args.num_folds > 1 else dict()
        self.filename_best = None
        self.best_epoch = 0
        self.early_stop = 0

    def learning(self, model, loaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.val_scores = checkpoint["val_scores"]
                self.best_epoch = checkpoint["best_epoch"]
                if "test_score" in checkpoint:
                    self.test_scores = checkpoint["test_scores"]
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint (val score: {})".format(checkpoint["val_score"]["Macro_AUC"]))
                if self.test_scores is not None:
                    print("=> loaded checkpoint (test score: {})".format(checkpoint["test_score"]["Macro_AUC"]))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            loader = loaders[-1]
            self.test_scores = self.validate(loader, model, criterion)
            return self.test_scores

        for epoch in range(self.best_epoch, self.args.num_epoch):
            self.epoch = epoch
            if self.args.num_folds > 1:
                train_loader, val_loader = loaders
            else:
                train_loader, val_loader, test_loader = loaders
            # train
            train_scores = self.train(train_loader, model, criterion, optimizer)
            # evaluate
            val_scores = self.validate(val_loader, model, criterion, status="val")
            is_best = (val_scores["Macro_AUC"] > self.val_scores["Macro_AUC"]) if self.val_scores is not None else True
            if self.args.num_folds > 1:
                if is_best:
                    self.val_scores = val_scores
                    self.best_epoch = self.epoch
                    self.save_checkpoint(
                        {
                            "best_epoch": self.best_epoch,
                            "state_dict": model.state_dict(),
                            "val_scores": self.val_scores,
                        }
                    )
            else:
                test_scores = self.validate(test_loader, model, criterion, status="test")
                if is_best:
                    self.val_scores = val_scores
                    self.test_scores = test_scores
                    self.best_epoch = self.epoch
                    self.save_checkpoint(
                        {
                            "best_epoch": self.best_epoch,
                            "state_dict": model.state_dict(),
                            "val_scores": self.val_scores,
                            "test_scores": self.test_scores,
                        }
                    )
            print(" *** best model {}".format(self.filename_best))
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")
            if is_best:
                self.early_stop = 0
            else:
                self.early_stop += 1
            if self.early_stop >= 10:
                print("Early stopping")
                break
        if self.args.num_folds > 1:
            return self.val_scores, self.best_epoch
        else:
            return self.val_scores, self.test_scores, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------train epoch {}-------------------------------".format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            logit = model(data_WSI)
            loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics
        scores, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        return scores

    def validate(self, data_loader, model, criterion, status="val"):
        model.eval()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="{} epoch {}".format(status, self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------{} epoch {}-------------------------------".format(status, self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            with torch.no_grad():
                logit = model(data_WSI)
                loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics
        if status == "val":
            scores, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        else:
            _, scores = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        return scores

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        if self.test_scores is not None:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{test_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    test_score=self.test_scores["Macro_AUC_mean"],
                    epoch=self.best_epoch,
                ),
            )
        else:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    epoch=self.best_epoch,
                ),
            )
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def metrics(self, logits, labels):
        general_meter = self.meter(num_classes=self.args.num_classes, bootstrap=False)
        bootstrapped_meter = self.meter(num_classes=self.args.num_classes, bootstrap=True)
        # general results
        general_results = general_meter(logits, labels)
        print("General Results:")
        print(
            "Macro AUC:    {:.4f},   Macro ACC:    {:.4f},   Macro F1:    {:.4f}".format(
                general_results["Macro_AUC"],
                general_results["Macro_ACC"],
                general_results["Macro_F1"],
            )
        )
        print(
            "Weighted AUC: {:.4f},   Weighted ACC: {:.4f},   Weighted F1: {:.4f}".format(
                general_results["Weighted_AUC"],
                general_results["Weighted_ACC"],
                general_results["Weighted_F1"],
            )
        )
        # bootstrapped results
        bootstrapped_results = bootstrapped_meter(logits, labels)
        print("Bootstrapped Results:")
        print(
            "Macro AUC:    {:.4f}±{:.4f},   Macro ACC:    {:.4f}±{:.4f},   Macro F1:    {:.4f}±{:.4f}".format(
                bootstrapped_results["Macro_AUC_mean"],
                bootstrapped_results["Macro_AUC_std"],
                bootstrapped_results["Macro_ACC_mean"],
                bootstrapped_results["Macro_ACC_std"],
                bootstrapped_results["Macro_F1_mean"],
                bootstrapped_results["Macro_F1_std"],
            )
        )
        print(
            "Weighted AUC: {:.4f}±{:.4f},   Weighted ACC: {:.4f}±{:.4f},   Weighted F1: {:.4f}±{:.4f}".format(
                bootstrapped_results["Weighted_AUC_mean"],
                bootstrapped_results["Weighted_AUC_std"],
                bootstrapped_results["Weighted_ACC_mean"],
                bootstrapped_results["Weighted_ACC_std"],
                bootstrapped_results["Weighted_F1_mean"],
                bootstrapped_results["Weighted_F1_std"],
            )
        )
        return general_results, bootstrapped_results

    def meter(self, num_classes, bootstrap=False):
        metrics: Dict[str, Metric] = {
            "Macro_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_F1": F1Score(num_classes=int(num_classes), average="macro", task="multiclass").to(self.device),
            "Macro_AUC": AUROC(num_classes=num_classes, average="macro", task="multiclass").to(self.device),
            "Weighted_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_F1": F1Score(num_classes=int(num_classes), average="weighted", task="multiclass").to(self.device),
            "Weighted_AUC": AUROC(num_classes=num_classes, average="weighted", task="multiclass").to(self.device),
        }
        # boot strap wrap
        if bootstrap:
            for k, m in metrics.items():
                # print("wrapping:", k)
                metrics[k] = BootStrapper(m, num_bootstraps=1000, sampling_strategy="multinomial").to(self.device)
        metrics = MetricCollection(metrics)
        return metrics
