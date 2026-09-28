# Inherently Interpretable Model

The implementation of inherently interpretable model employment and evaluation is based on [LaBo](https://github.com/YueYANG1996/LaBo), with ConceptCLIP as the backbone CLIP model. We use WBCAtt here (requires one GPU with 24GB memory).

Obtain the [WBCAtt attribute annotations and preparation instructions](https://github.com/apple2373/wbcatt/tree/main/submission) from its authors and the [PBC source images](https://data.mendeley.com/datasets/snkd93bnjr/1) from the original dataset release. The PBC image download alone does not contain the WBCAtt attribute annotations. Follow each provider's terms and cite both datasets.

Prepare local images under `data/images`, preserving the paths required by the supplied splits in `data/splits`. Prepare the concept files following LaBo and set their locations in [cfg/asso_opt/WBCAtt/WBCAtt_base.py](./cfg/asso_opt/WBCAtt/WBCAtt_base.py); its default concept directory is `data/concepts`.

``` bash
# Inherently interpretable model employment and evaluation
cd downstream_evaluation/inherently_interpretable_model
python main.py \
    --cfg cfg/asso_opt/WBCAtt/WBCAtt_allshot_fac.py \
    --work-dir data/checkpoint \
    --func asso_opt_main

```
