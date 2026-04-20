import torchmetrics
import wandb
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from data import DataModule, DotProductDataModule
import sys
sys.path.append("../..")
import json


class AssoConcept(pl.LightningModule):
    def init_weight_concept(self, concept2cls):
        # the initial weight of the class-concept weight matrix (n_cls, n_concept) 
        self.init_weight = th.zeros((self.cfg.num_cls, len(self.select_idx))) 

        if self.cfg.use_rand_init: th.nn.init.kaiming_normal_(self.init_weight)
        else: self.init_weight.scatter_(0, concept2cls, self.cfg.init_val) 
            
        if 'cls_name_init' in self.cfg and self.cfg.cls_name_init != 'none':
            if self.cfg.cls_name_init == 'replace':
                self.init_weight = th.load(self.init_weight_save_dir)
            elif self.cfg.cls_name_init == 'combine':
                self.init_weight += th.load(self.init_weight_save_dir)
                self.init_weight = self.init_weight.clip(max=1)
            elif self.cfg.cls_name_init == 'random':
                th.nn.init.kaiming_normal_(self.init_weight)


    def __init__(self, cfg, init_weight=None, select_idx=None, data_module=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')
        self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        cls_sim_path = self.data_root.joinpath('cls_sim.pth')

        if not concept_feat_path.exists():
            raise RuntimeError('need to call datamodule precompute_txt before using the model')
        else:
            if select_idx is None: self.select_idx = th.load(select_idx_path)[:cfg.num_concept]  # the ids of selected concepts
            else: self.select_idx = select_idx

            self.concepts = th.load(concept_feat_path)[self.select_idx].cuda()  # the selected concept features
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            self.concept_raw = np.load(concept_raw_path)[self.select_idx]  # array of raw concepts
            self.concept2cls = th.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1) # class of the concepts

        if init_weight is None:
            self.init_weight_concept(self.concept2cls)  # initalize the class-concept weight matrix (n_cls, n_concept)
        else:
            self.init_weight = init_weight

        if 'cls_sim_prior' in self.cfg and self.cfg.cls_sim_prior and self.cfg.cls_sim_prior != 'none':  # 'none'
            # class similarity is prior to restrict class-concept association
            # if class A and B are dissimilar (similarity=0), then the mask location will be 0 
            print('use cls prior')
            cls_sim = th.load(cls_sim_path)
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                target_class = int(th.where(self.init_weight[:,concept_id] == 1)[0])
                new_weights.append(cls_sim[target_class] + self.init_weight[:,concept_id])
            self.init_weight = th.vstack(new_weights).T
            # self.weight_mask = cls_sim @ self.init_weight

        self.asso_mat = th.nn.Parameter(self.init_weight.clone())  # asso_mat is the class-concept weight matrix
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass") 
        self.valid_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass")
        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass")
        self.all_y = []
        self.all_pred = []
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.cfg.num_cls, task="multiclass")
        self.save_hyperparameters()  
        self.ci_results = [] 
        self.max_val_aucs = []  

        if data_module != None:
            self.data_module = data_module

    def _get_weight_mat(self):
        if self.cfg.asso_act == 'relu':
            mat = F.relu(self.asso_mat)
        elif self.cfg.asso_act == 'tanh':
            mat = F.tanh(self.asso_mat) 
        elif self.cfg.asso_act == 'softmax':
            mat = F.softmax(self.asso_mat, dim=-1) 
        else:
            mat = self.asso_mat
        return mat 

    # TODO: please see the forward function of AssoConceptFast in the bottom
    def forward(self, img_feat):
        mat = self._get_weight_mat()  # the class-concept weight matrix (n_cls, n_concept)
        cls_feat = mat @ self.concepts  # the class features (text features) (the multiplication of the weight matrix and the concept features)
        sim = img_feat @ cls_feat.t()  # the image-text (class) similarity
        return sim


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        sim = self.forward(image)  # (batch_size, n_cls)
            
        if self.cfg.clip_model == "ConceptCLIP":
            pred = 96.6067 * sim  # logit scale
        else:
            pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace()

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1, dim=-1).max() #.mean()

        self.train_acc(pred, label)
        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss
    
    # TODO: new add to test at each training epoch
    def training_epoch_end(self, outputs):

        all_preds = []
        all_labels = []
        all_logits = []
        for batch in self.data_module.test_dataloader():
            image, y = batch
            sim = self.forward(image)
            if self.cfg.clip_model == "ConceptCLIP":
                pred = 96.6067 * sim
            else:
                pred = 100 * sim  # scaling as standard CLIP does
            pred = pred.cpu()
            y = y.cpu()
            y_logits = th.softmax(pred, dim=-1)
            y_pred = pred.argmax(dim=-1)
            all_preds.append(y_pred)
            all_labels.append(y)
            all_logits.append(y_logits)

        all_preds = th.hstack(all_preds)        
        all_labels = th.hstack(all_labels)      
        acc = self.test_acc(all_preds, all_labels)
        
        all_logits = th.cat(all_logits, dim=0) 
        all_labels = all_labels.detach().numpy()
        all_logits = all_logits.detach().numpy()

        if max(all_labels) > 1:  # multi-class
            auc_ovr = roc_auc_score(all_labels, all_logits, multi_class="ovr")
            self.log("test_auc", auc_ovr, on_step=False, on_epoch=True)
            auc = auc_ovr
        
        else:  # binary class
            all_labels_new = []
            for label in all_labels:
                if label == 1:
                    label = [0,1]
                else:
                    label = [1,0]
                all_labels_new.append(label)

            auc = roc_auc_score(all_labels_new, all_logits)
            self.log("test_auc", auc, on_step=False, on_epoch=True)

        if self.current_epoch == self.cfg.max_epochs - 1:  # record the result of last epoch
            self.log("test_auc_last_epoch", auc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt


    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        # self.log('val_loss', loss)
        self.valid_acc(pred, y)
        # self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        
        if self.cfg.clip_model == "ConceptCLIP":
            pred = 96.6067 * sim
        else:
            pred = 100 * sim
        pred = pred.cpu()
        y = y.cpu()
        y_logits = th.softmax(pred, dim=-1)
        y_pred = pred.argmax(dim=-1)

        self.all_y.append(y)
        self.all_pred.append(y_logits)

    def predict_step(self, batch, batch_idx):
        image, y, image_name = batch
        sim = self.forward(image)
        pred = 100 * sim
        _, y_pred = th.topk(pred, self.num_pred)
        for img_path, gt, top_pred in zip(image_name, y, y_pred):
            gt = (gt + 1).item()
            top_pred = (top_pred + 1).tolist()
            self.pred_table.add_data(wandb.Image(img_path), gt, *top_pred)
    

    def on_predict_epoch_end(self, results):
        wandb.log({'pred_table': self.pred_table})
    

    def prune_asso_mat(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat().detach()
        val_asso_mat = th.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = th.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good


    def extract_cls2concept(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat().detach()
        strong_asso = asso_mat > thresh 
        res = {}
        import pdb; pdb.set_trace()
        for i, cls_name in enumerate(cls_names): 
            ## threshold globally
            keep_idx = strong_asso[i]
            ## sort
            res[cls_name] = np.unique(self.concept_raw[keep_idx])
        return res


    def extract_concept2cls(self, percent_thresh=0.95, mode='global'):
        asso_mat = self.asso_mat.detach()
        res = {} 
        for i in range(asso_mat.shape[1]):
            res[i] = th.argsort(asso_mat[:, i], descending=True).tolist()
        return res


class AssoConceptFast(AssoConcept):

    def forward(self, dot_product):
        mat = self._get_weight_mat()
        return dot_product @ mat.t() 