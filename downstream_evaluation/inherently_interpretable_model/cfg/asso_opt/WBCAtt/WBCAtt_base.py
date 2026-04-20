# TODO: please set the correct path in this file

_base_ = '../base.py'

clip_model = 'ConceptCLIP'

# dataset
proj_name = "WBCAtt_" + clip_model

# path that saves class2concepts.json (concepts_raw.npy, concept2cls.npy, cls_names.npy)
concept_root = 'data/concepts/'

# path that saves class2images_train/val/test.p 
img_split_path = 'data/splits/'

img_path = 'data/images'

concept_type = "all"
img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 5 

## data loader
bs = 32
on_gpu = True

# concept select
num_concept = 23 

# weight matrix fitting
lr = 1e-4
max_epochs = 500

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

