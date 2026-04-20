"""Adapted from: https://github.com/YueYANG1996/LaBo """

import os
import utils
import json
import argparse
import pytorch_lightning as pl
import torch as th
import numpy as np
from sklearn.metrics import confusion_matrix
import random

def save_npy_files(class2concepts, save_dir):
    # sort class name to make sure they are in the same order, to avoid potential problem
    class_names = sorted(list(class2concepts.keys()))
    num_concept = sum([len(concepts) for concepts in class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    i = 0
    all_concepts = []
    for class_name, concepts in class2concepts.items():
        class_idx = class_names.index(class_name)
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx
            i += 1
    np.save(save_dir + 'concepts_raw.npy', np.array(all_concepts))
    np.save(save_dir + 'cls_names.npy', np.array(class_names))
    np.save(save_dir + 'concept2cls.npy', concept2cls)


def asso_opt_main(cfg):
    from models.asso_opt.asso_opt import AssoConcept, AssoConceptFast
    from models.select_concept.select_algo import mi_select, clip_score_select, group_mi_select, group_clip_select, submodular_select, random_select
    from data import DataModule, DotProductDataModule
    import random
    proj_name = cfg.proj_name  # e.g., "WBCAtt"

    if cfg.class2concepts is not None: 
        class2concepts = cfg.class2concepts
    else:
        class2concepts = json.load(open(cfg.concept_root + "class2concepts.json", "r"))  # class2concepts: a json file contains {"class": ["concepts"]}
    print("class2concepts: ", class2concepts)
    save_npy_files(class2concepts, cfg.concept_root)  # save an npy file for class2concepts
    
    # concept seletion method
    if cfg.concept_select_fn == "pre_defined":  # pre-defined concept list
        concept_select_fn = None
        print("use pre-defined concepts")
    elif cfg.concept_select_fn == "submodular":
        concept_select_fn = submodular_select
        print("use submodular")
    elif cfg.concept_select_fn == "random":
        concept_select_fn = random_select
        print("use random")
    else:
        if cfg.use_mi:
            if cfg.group_select:
                concept_select_fn = group_mi_select
                print("use grounp mi")
            else:
                concept_select_fn = mi_select
                print("use mi")
        else:
            if cfg.group_select:
                concept_select_fn = group_clip_select
                print("use group clip")
            else:
                concept_select_fn = clip_score_select
                print("use clip")
    
    random.seed(1)

    try: print("cfg.submodular_weights: ", cfg.submodular_weights)
    except: cfg.submodular_weights = "none"

    print("use dot product dataloader")

    data_module = DotProductDataModule(  
        cfg.num_concept,
        cfg.data_root,
        cfg.clip_model,
        cfg.img_split_path,
        cfg.img_path,
        cfg.n_shots,
        cfg.raw_sen_path,
        cfg.concept2cls_path,
        concept_select_fn,
        cfg.cls_name_path,
        cfg.bs,
        on_gpu=cfg.on_gpu,
        num_workers=cfg.num_workers if 'num_workers' in cfg else 0,
        img_ext=cfg.img_ext if 'img_ext' in cfg else '.jpg',
        clip_ckpt=cfg.ckpt_path if 'ckpt_path' in cfg else None,
        use_txt_norm=cfg.use_txt_norm if 'use_txt_norm' in cfg else False, 
        use_img_norm=cfg.use_img_norm if 'use_img_norm' in cfg else False,
        use_cls_name_init=cfg.cls_name_init if 'cls_name_init' in cfg else 'none',
        use_cls_sim_prior=cfg.cls_sim_prior if 'cls_sim_prior' in cfg else 'none',
        remove_cls_name=cfg.remove_cls_name if 'remove_cls_name' in cfg else True,
        submodular_weights=cfg.submodular_weights,
        )


    print("use asso concept with dot product loader, faster") 
    model = AssoConceptFast(cfg, init_weight=th.load(cfg.init_weight_path) if 'init_weight_path' in cfg else None, data_module=data_module)

    check_interval = 10
    print("check interval = {}".format(check_interval))

    if not cfg.DEBUG:
        if 'use_last_ckpt' in cfg and cfg['use_last_ckpt']:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{val_acc:.4f}',
                every_n_epochs=check_interval)
        else:  
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{test_auc:.4f}', 
                monitor = 'test_auc',
                mode='max',
                save_top_k=-1,  
                every_n_epochs=100) 
        
        wandb_logger = pl.loggers.WandbLogger(name='{}_{}shot_{}_{}'\
                       .format(cfg.proj_name, cfg.n_shots, cfg.concept_type, cfg.num_concept),
                       project="LaBo",
                       config=cfg._cfg_dict)

        trainer = pl.Trainer(gpus=1,
                             callbacks=[checkpoint_callback],
                             logger=wandb_logger,
                             check_val_every_n_epoch=check_interval,
                             max_epochs=cfg.max_epochs if 'max_epochs' in cfg else 1000)
    else:
        if 'use_last_ckpt' in cfg and cfg['use_last_ckpt']:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{val_acc:.4f}',
                every_n_epochs=check_interval)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=cfg.work_dir,
            filename='{epoch}-{step}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            every_n_epochs=50)
        # device_stats = pl.callbacks.DeviceStatsMonitor()
        trainer = pl.Trainer(gpus=1, max_epochs=1000, \
            callbacks=[checkpoint_callback, ], check_val_every_n_epoch=50, default_root_dir=cfg.work_dir)
    
    if cfg.resume and cfg.model_ckpt is not None:
        print(f"resume the model from {cfg.model_ckpt}.")
        trainer.fit(model, data_module, ckpt_path=cfg.model_ckpt)
    else:
        print("train from scratch.")
        trainer.fit(model, data_module)