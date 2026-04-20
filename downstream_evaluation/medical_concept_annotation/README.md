# Medical Concept Annotation


Using Derm7pt dataset (requires one GPU with 24GB memory):

``` bash
# Please put the downloaded dataset images to this directory
# ./downstream_evaluation/medical_concept_annotation/data/images

# Zero-shot medical concept annotation
cd downstream_evaluation/medical_concept_annotation
python concept_annotation.py

```