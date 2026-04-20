from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch import Tensor, nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import ModelOutput
from transformers import PreTrainedModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class Pooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@dataclass(init=False)
class VQAOutput(ModelOutput):
    logits: torch.FloatTensor
    image_pooler_output: Optional[torch.FloatTensor]
    text_pooler_output: Optional[torch.FloatTensor]


class VQAModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_answer_labels: int,
        text_embed_dim: int = 1152,
        image_embed_dim: int = 1152,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        max_position_embeddings: int = 50,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        # cross modal linear transform
        self.cross_modal_text_transform = nn.Linear(text_embed_dim, hidden_size)
        self.cross_modal_image_transform = nn.Linear(image_embed_dim, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * mlp_ratio,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            is_decoder=True,
            add_cross_attention=True,
        )

        # cross modal attention layers
        self.cross_modal_image_layers = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(num_hidden_layers)]
        )
        self.cross_modal_text_layers = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(num_hidden_layers)]
        )

        # cross modal pooler
        self.cross_modal_image_pooler = Pooler(hidden_size=hidden_size)
        self.cross_modal_text_pooler = Pooler(hidden_size=hidden_size)

        # classifier for VQA
        self.vqa_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_answer_labels),
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        text_extend_attention_masks: torch.FloatTensor,
        image_extend_attention_masks: torch.FloatTensor,
        return_dict: bool = False,
    ) -> Union[Tuple, VQAOutput]:
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros(text_embeds.shape[0], text_embeds.shape[1], dtype=torch.long).cuda()
        )

        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.ones(image_embeds.shape[0], image_embeds.shape[1], dtype=torch.long).cuda()
        )

        text_features, image_features = (text_embeds, image_embeds)

        # cross modal layers
        for text_layer, image_layer in zip(
            self.cross_modal_text_layers, self.cross_modal_image_layers
        ):
            text_features = text_layer(
                hidden_states=text_features,
                attention_mask=text_extend_attention_masks,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_extend_attention_masks,
            )[0]
            image_features = image_layer(
                hidden_states=image_features,
                attention_mask=image_extend_attention_masks,
                encoder_hidden_states=text_features,
                encoder_attention_mask=text_extend_attention_masks,
            )[0]

        text_pooler_output = self.cross_modal_text_pooler(text_features)
        image_pooler_output = self.cross_modal_image_pooler(image_features)

        pooler_output = torch.cat([text_pooler_output, image_pooler_output], dim=-1)

        # Classification for VQA
        logits = self.vqa_classifier(pooler_output)

        if not return_dict:
            return (logits, image_pooler_output, text_pooler_output)

        return VQAOutput(
            logits=logits,
            image_pooler_output=image_pooler_output,
            text_pooler_output=text_pooler_output,
        )


class CompoundModel(torch.nn.Module):
    """
    Wrapper for the vqa model
    """

    def __init__(
        self,
        model,
        vqa_model: VQAModel,
    ):
        super().__init__()

        self.model = model.cuda()
        self.vqa_model = vqa_model.cuda()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        **kwargs
    ):
        # input_ids: [32, 50]
        # pixel_values: [32, 3, 224, 224]
        self.model = self.model.cuda()
        
        res = self.model(pixel_values, input_ids)
        image_embeds = res["image_token_features"]
        text_embeds = res["text_token_features"]
        # image_embeds: [B, 729, 1152]  <class 'torch.Tensor'>
        # text_embeds: [B, 50, 1152]  <class 'torch.Tensor'>
        text_extend_attention_masks = self.get_extended_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            dtype=torch.float,
        )
        # text_extend_attention_masks: [B, 1, 1, 50]  <class 'torch.Tensor'>
        image_mask = torch.ones(
            (image_embeds.shape[0], image_embeds.shape[1]),
            dtype=torch.long,
            device=image_embeds.device,
        )
        # image_mask: [B, 729]  <class 'torch.Tensor'>
        image_extend_attention_masks = self.get_extended_attention_mask(
            attention_mask=image_mask, input_shape=image_mask.shape, dtype=torch.float
        )
        # image_extend_attention_masks: [B, 1, 1, 729]  <class 'torch.Tensor'>
        vqa_output: VQAOutput = self.vqa_model(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_extend_attention_masks=text_extend_attention_masks,
            image_extend_attention_masks=image_extend_attention_masks,
            return_dict=True,
        )
        # for key in vqa_output.keys():
            # print(key, vqa_output[key].shape, type(vqa_output[key]))
        # logits: [B, 497]  <class 'torch.Tensor'>
        # image_pooler_output: [B, 768]  <class 'torch.Tensor'>
        # text_pooler_output: [B, 768]  <class 'torch.Tensor'>
        return vqa_output

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask
