# Pathology Whole-Slide Image Analysis

## Feature Extraction

We follow the standard process in https://github.com/mahmoodlab/CLAM.

Please use the following code to load the model.

```python
def get_conceptclip_model(device):
    # Initialize model and processor
    model = AutoModel.from_pretrained('../../pre_training/src/pretrained_checkpoint/ConceptCLIP', trust_remote_code=True)
    model_name = 'ConceptCLIP'
    model = model.to(device)
    original_forward = model.forward

    def patched_forward(self, image=None, text=None, **kwargs):
        return original_forward(
            pixel_values=image,
            input_ids=text,
            **kwargs
        )

    model.forward = types.MethodType(patched_forward, model)
    model.eval()
    def func(image):
        # get the features
        with torch.no_grad():
            image_embs, _ = model.encode_image(image, normalize=False)
            return image_embs
        
    return func


def get_conceptclip_trans():
    preprocess_cfg = {'interpolation': 'bicubic',
    'mean': [0.5, 0.5, 0.5],
    'resize_mode': 'squash',
    'size': 384,
    'std': [0.5, 0.5, 0.5]}
    pp_cfg = PreprocessCfg(**preprocess_cfg)
    preprocess = image_transform_v2(pp_cfg, is_train=False)

    return preprocess
```

You will get a folder like this:
```bash
output
  └─example
    └─pt_files
      └─conceptclip
        ├── example_1.pt
        ├── example_2.pt
        └── example_3.pt
```

## Downstream Tasks

Once you get the feature you can train an ABMIL model to conduct downstream tasks.

Set the `ROOT_FEATURE` in the `scripts/train.sh` based on your feature folder and you can run the scripts to train the model.

