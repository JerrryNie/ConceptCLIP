_base_ = 'WBCAtt_base.py'
n_shots = "all"

data_root = 'data/checkpoint/WBCAtt_allshot_fac'  # TODO: please set the path to save outputs
lr = 5e-4
bs = 256

concept_type = "pre_defined"
concept_select_fn = "pre_defined"
submodular_weights = None

class2concepts = None
