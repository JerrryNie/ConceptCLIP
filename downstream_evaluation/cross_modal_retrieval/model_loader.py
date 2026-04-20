import torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoConfig
from open_clip.transform import PreprocessCfg, image_transform_v2
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
import types

def load_biomedclip():
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess_val = create_model_from_pretrained(model_name)
    # model = model.visual
    preprocess_train = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=True,
    )
    tokenizer = get_tokenizer(model_name)
    output_dim = model.visual.head.proj.out_features
    model_name = model_name.replace('/', '_')
    
    return model_name, model, preprocess_train, preprocess_val, tokenizer, output_dim


def load_siglip_400m():
    model_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
    model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, output_dict=True)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = get_tokenizer(model_name)
    output_dim = model.visual.trunk.attn_pool.mlp.fc2.out_features
    model_name = model_name.replace('/', '_')
    
    return model_name, model, preprocess_train, preprocess_val, tokenizer, output_dim


def load_clip_b():
    model_name = 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K'
    model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, output_dict=True)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = get_tokenizer(model_name)
    model.visual.output_tokens = True
    model_name = model_name.replace('/', '_')
    output_dim = model.ln_final.weight.shape[0]
    
    return model_name, model, preprocess_train, preprocess_val, tokenizer, output_dim


def load_conceptclip():
    model_name = 'ConceptCLIP'
    model = AutoModel.from_pretrained('../../pre_training/src/pretrained_checkpoint/ConceptCLIP', trust_remote_code=True)
    original_forward = model.forward

    def patched_forward(self, image=None, text=None, **kwargs):
        return original_forward(
            pixel_values=image,
            input_ids=text,
            **kwargs
        )

    model.forward = types.MethodType(patched_forward, model)

    preprocess_cfg = {'interpolation': 'bicubic',
        'mean': [0.5, 0.5, 0.5],
        'resize_mode': 'squash',
        'size': 384,
        'std': [0.5, 0.5, 0.5]
    }
    pp_cfg = PreprocessCfg(**preprocess_cfg)
    preprocess_train = image_transform_v2(pp_cfg, is_train=True)
    preprocess_val = image_transform_v2(pp_cfg, is_train=False)
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    model.eval()
    model_name = model_name.replace('/', '_')
    output_dim = model.image_proj[2].out_features
    
    return model_name, model, preprocess_train, preprocess_val, tokenizer, output_dim


def load_pmc_clip():
    from collections import OrderedDict
    from typing import Optional, Sequence, Tuple
    import math

    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

    from transformers import AutoTokenizer, AutoModel

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1):
            super().__init__()

            # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.relu2 = nn.ReLU(inplace=True)

            self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

            self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu3 = nn.ReLU(inplace=True)

            self.downsample = None
            self.stride = stride

            if stride > 1 or inplanes != planes * Bottleneck.expansion:
                # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
                self.downsample = nn.Sequential(OrderedDict([
                    ("-1", nn.AvgPool2d(stride)),
                    ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                    ("1", nn.BatchNorm2d(planes * self.expansion))
                ]))

        def forward(self, x: torch.Tensor):
            identity = x

            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.relu2(self.bn2(self.conv2(out)))
            out = self.avgpool(out)
            out = self.bn3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu3(out)
            return out

    class AttentionPool2d(nn.Module):
        def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
            super().__init__()
            self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
            self.num_heads = num_heads

        def forward(self, x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            # print(f'x.shape: {x.shape}')
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
            )
            # print(f'x.shape: {x.shape}')
            return x[0], x[1:].transpose(0, 1)

    class ModifiedResNet(nn.Module):
        """
        A ResNet class that is similar to torchvision's but contains the following changes:
        - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
        - The final pooling layer is a QKV attention instead of an average pool
        """

        def __init__(self, layers, output_dim, heads, image_size=224, width=64):
            super().__init__()
            self.output_dim = output_dim
            self.image_size = image_size

            # the 3-layer stem
            self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width // 2)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(width // 2)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(width)
            self.relu3 = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(2)

            # residual layers
            self._inplanes = width  # this is a *mutable* variable used during construction
            self.layer1 = self._make_layer(width, layers[0])
            self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
            self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

            self.init_parameters()

        def _make_layer(self, planes, blocks, stride=1):
            layers = [Bottleneck(self._inplanes, planes, stride)]

            self._inplanes = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                layers.append(Bottleneck(self._inplanes, planes))

            return nn.Sequential(*layers)

        def init_parameters(self):
            if self.attnpool is not None:
                std = self.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        def lock(self, unlocked_groups=0, freeze_bn_stats=False):
            assert unlocked_groups == 0, 'partial locking not currently supported for this model'
            for param in self.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self)

        @torch.jit.ignore
        def set_grad_checkpointing(self, enable=True):
            # FIXME support for non-transformer
            pass

        def stem(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x_cls, x_tokens = self.attnpool(x)

            visual_output = dict.fromkeys(["image_features", "mim_loss"], None)
            visual_output.update({
                'image_features': x_cls,
                'image_token_features': x_tokens
            })

            return visual_output

    # Image preprocess

    def _convert_to_rgb(image):
        return image.convert('RGB')

    def image_transform(
            image_size: int,
            mean: Optional[Tuple[float, ...]] = None,
            std: Optional[Tuple[float, ...]] = None,
            fill_color: int = 0,
    ):
        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        normalize = Normalize(mean=mean, std=std)

        transforms = [
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            # CenterCrop(image_size),
        ]
        transforms.extend([
            # _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

    def image_transform_train(
            image_size: int,
            mean: Optional[Tuple[float, ...]] = None,
            std: Optional[Tuple[float, ...]] = None,
            fill_color: int = 0,
    ):
        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        normalize = Normalize(mean=mean, std=std)

        transforms = [
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
        transforms.extend([
            # _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

    class PMC_CLIP(nn.Module):
        def __init__(
                self,
        ):
            super().__init__()

            self.context_length = 77

            # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
            # memory efficient in recent PyTorch releases (>= 1.10).
            # NOTE: timm models always use native GELU regardless of quick_gelu flag.

            self.visual = ModifiedResNet(
                layers=[3,4,6,3],
                output_dim=768,
                heads=32,
                image_size=224,
                width=64
            )

            # Tokenizer
            tokenizer_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
            self.cls_id = 2  # [CLS]'s token id is 2, while it varies from tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            self.text_encoder = AutoModel.from_pretrained(tokenizer_name)#, return_dict=True)

            self.text_projection = nn.Parameter(torch.empty(768, 768))
            self.softmax = nn.LogSoftmax(dim=-1)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.img_special_token = nn.Parameter(torch.zeros(1, 1, 768))

        def _encode_image(self, image):
            return self.visual(image)
        
        def encode_image(self, image):
            raw_image_features = self._encode_image(image)
            image_features = F.normalize(raw_image_features['image_features'], dim=-1)  # [128, 768]
            return image_features

        def encode_text(self, texts, normalize=True):
            x = self.text_encoder(
                input_ids = texts,
                output_attentions = False
            )
            x = x['last_hidden_state']
            last_token_index = torch.nonzero( (texts == self.cls_id).squeeze() )
            text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]
            text_features = text_features @ self.text_projection  # NOTE for matching

            # text_output = dict.fromkeys(["text_features"], None)
            # for key in text_output:
            #     text_output[key] = eval(key)  # HACK dark magic, could be dangerous
            if normalize:
                text_features = F.normalize(text_features, dim=-1)

            return text_features

        def forward(self, images=None, texts=None):
            # print(f'images: {images}')
            if images is not None:
                raw_image_features = self._encode_image(images)
                # print(f'image_features after self.encode_image: {image_features}')
                image_features = F.normalize(raw_image_features['image_features'], dim=-1)  # [128, 768]
                image_token_features = raw_image_features['image_token_features']
            else:
                image_features = None

            if texts is not None:
                text_features = self.encode_text(texts)
                text_features = F.normalize(text_features, dim=-1)
            else:
                text_features = None

            clip_prediction = dict.fromkeys(["image_features", "image_token_features", "text_features", "logit_scale"], None)
            clip_prediction.update({
                "image_features": image_features,
                "image_token_features": image_token_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp() if self.logit_scale is not None else None,
            })
            return clip_prediction

    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    tokenizer.context_length = 77
    preprocess_val = image_transform(
        image_size=224,
    )
    preprocess_train = image_transform_train(
        image_size=224,
    )
    model = PMC_CLIP()
    ckpt_path = '/data/ynieae/ckpts/pmc-clip/checkpoint.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('module.'):
            name = k[7:]
            new_ckpt[name] = v
    incompatible_keys = model.load_state_dict(new_ckpt, strict=False)
    model.eval()
    model_name = 'PMC_CLIP'
    model_name = model_name.replace('/', '_')
    out_dim = model.visual.attnpool.c_proj.out_features

    return model_name, model, preprocess_train, preprocess_val, tokenizer, out_dim


MODELS = {
    'clip_b': load_clip_b,
    'siglip_400m': load_siglip_400m,
    'biomedclip': load_biomedclip,
    'pmc_clip': load_pmc_clip,
    'conceptclip': load_conceptclip,
}
