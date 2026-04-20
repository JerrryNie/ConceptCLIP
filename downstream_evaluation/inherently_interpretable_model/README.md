# Inherently Interpretable Model

The implementation of inherently interpretable model employment and evaluation is based on [LaBo](https://github.com/YueYANG1996/LaBo), with the proposed ConceptCLIP as the backbone CLIP model. We use WBCAtt dataset here and please put the downloaded images and related files (e.g., `class2concepts.json`, can be generated using the pickle.dump() function, please refer to LaBo) under the `downstream_evaluation/inherently_interpretable_model/data` folder (requires one GPU with 24GB memory).

``` bash
# Inherently interpretable model employment and evaluation
cd downstream_evaluation/inherently_interpretable_model
python main.py \
    --cfg cfg/asso_opt/WBCAtt/WBCAtt_allshot_fac.py \
    --work-dir data/checkpoint \
    --func asso_opt_main

```