# Medical Concept Annotation


Using Derm7pt dataset (requires one GPU with 24GB memory):

Obtain the images and metadata from the [official Derm7pt download page](https://derm.cs.sfu.ca/Download.html), completing the provider's password request and following its dataset license. Place the images under `data/images` relative to this task directory, retaining the paths listed in the `derm` column of the supplied [test metadata](./data/meta/meta_test.csv).

``` bash
# Please put the downloaded dataset images to this directory
# ./downstream_evaluation/medical_concept_annotation/data/images

# Zero-shot medical concept annotation
cd downstream_evaluation/medical_concept_annotation
python concept_annotation.py

```
