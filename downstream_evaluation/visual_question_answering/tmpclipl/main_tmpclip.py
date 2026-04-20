import logging
import os
import os.path
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils
from datasets import load_dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter.*")
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    get_polynomial_decay_schedule_with_warmup,
)
import types
from torchvision.transforms import Normalize, InterpolationMode, Resize, CenterCrop
from collections import OrderedDict
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers.optimization import AdamW

from open_clip.transform import PreprocessCfg, image_transform_v2

from model_tmpclip import VQAModel, CompoundModel, VQAOutput

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
        default=None,
        metadata={
            "help": "Loading script of the dataset. Could be either `vqa-rad.py` or `slake.py`"
        },
    )

    cache_dir: Optional[str] = field(
        default="data/cache",
        metadata={"help": "Cache directory for loading datasets."},
    )

    train_val_split: Optional[float] = field(
        default=0.1, metadata={"help": "Percent to split off of train for validation."}
    )

    max_seq_length: Optional[int] = field(
        default=50,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker predict, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    image_column: Optional[str] = field(
        default="image_path",
        metadata={
            "help": "The name of the column in the datasets containing the full image file paths."
        },
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={
            "help": "The name of the column in the datasets containing the visual questions."
        },
    )

    def __post_init__(self):
        pass


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        default="hf",
        metadata={
            "help": "Type of the pretrained clip model, can be either `hf` or `open_clip`"
        },
    )

    text_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained text model"},
    )

    vision_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained vision model"},
    )

    vqa_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained vqa model"},
    )

    freeze_vision_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision model parameters or not."},
    )
    freeze_text_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the text model parameters or not."},
    )

    num_answer_labels: int = field(
        default=None,
        metadata={"help": "Number of answers"},
    )

    hidden_size: int = field(
        default=768,
        metadata={
            "help": "Dimensionality of the crossmodal encoder layers and the pooler layer."
        },
    )

    num_hidden_layers: int = field(
        default=6,
        metadata={
            "help": "Number of hidden layers in the Transformer encoder. Ignore this if you are under hf framework."
        },
    )

    num_attention_heads: int = field(
        default=12,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder."
        },
    )

    mlp_ratio: int = field(
        default=4,
        metadata={
            "help": "Dimensionality ratio of the `intermediate` (i.e., feed-forward layer) in the Transformer encoder."
        },
    )

    dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability of the Transformer encoder layer."},
    )


dataset_name_mapping = {
    "vqa-rad.py": ("image_path", "question", "label", "type"),
    "slake.py": ("image_path", "question", "label", "type"),
}


class Transform(torch.nn.Module):
    def __init__(self, image_size: int, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )

    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    types = torch.tensor([example["type"] for example in examples], dtype=torch.int32)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "types": types,
        "return_loss": True,
    }


def get_optimizer(
    model: CompoundModel,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_ratio: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
) -> Tuple:
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = [
        "vqa_classifier",
        "nlvr2_classifier",
        "mlm_score",
        "itm_score",
        "snli_classifier",
    ]
    cross_modal_names = ["cross_modal"]
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    end_lr = 0.0

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate * lr_mult_cross_modal,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate * lr_mult_cross_modal,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=1e-8,
        betas=(adam_beta1, adam_beta2),
    )

    warmup_steps = int(warmup_ratio * max_steps)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        lr_end=end_lr,
        power=1,
    )

    return (optimizer, scheduler)


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        print(f"Loading arguments in {sys.argv[1]}")
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset

    dataset = load_dataset(
        data_args.dataset_name, trust_remote_code=True, cache_dir=data_args.cache_dir
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = (
        None if "validation" in dataset.keys() else data_args.train_val_split
    )
    if (
        isinstance(data_args.train_val_split, float)
        and data_args.train_val_split > 0.0
        and training_args.do_train
    ):
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # 5. Load pretrained model, tokenizer, and image processor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    print("Vocab size:", tokenizer.vocab_size)
    tokenizer.context_length = 77
    print("Context length:", tokenizer.context_length)
    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    config = AutoConfig.from_pretrained(model_args.vision_model_path, trust_remote_code=True)
    if 'conceptclip' not in model_args.vision_model_path.lower():
        pp_cfg = PreprocessCfg(**config.preprocess_cfg)
        image_processor = image_transform_v2(pp_cfg, is_train=False)
        model = AutoModel.from_pretrained(model_args.vision_model_path, trust_remote_code=True)
    else:
        preprocess_cfg = {'interpolation': 'bicubic',
            'mean': [0.5, 0.5, 0.5],
            'resize_mode': 'squash',
            'size': 384,
            'std': [0.5, 0.5, 0.5]
        }
        pp_cfg = PreprocessCfg(**preprocess_cfg)
        image_processor = image_transform_v2(pp_cfg, is_train=False)
        model = AutoModel.from_pretrained(model_args.vision_model_path, trust_remote_code=True)
        original_forward = model.forward
        def patched_forward(self, image=None, text=None, **kwargs):
            return original_forward(
                pixel_values=image,
                input_ids=text,
                **kwargs
            )

        model.forward = types.MethodType(patched_forward, model)
    

    model.requires_grad_(not model_args.freeze_vision_model)
    # load vqa model
    if model_args.vqa_model_path is not None:
        vqa_model = VQAModel.from_pretrained(model_args.vqa_model_path)
    else:
        vqa_model = VQAModel(
            num_answer_labels=model_args.num_answer_labels,
            text_embed_dim=config.embed_dim,
            image_embed_dim=config.embed_dim,
            hidden_size=model_args.hidden_size,
            vocab_size=tokenizer.vocab_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            mlp_ratio=model_args.mlp_ratio,
            max_position_embeddings=data_args.max_seq_length,
            dropout_prob=model_args.dropout_prob,
        )
    model = CompoundModel(
        model=model, vqa_model=vqa_model
    )

    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.question_column is None:
        question_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        384,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    )
    image_transformations = torch.jit.script(image_transformations)

    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        questions = list(examples[question_column])
        text_inputs = tokenizer(
            questions,
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = [
            read_image(image_file, mode=ImageReadMode.RGB)
            for image_file in examples[image_column]
        ]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if not col in dataset_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if not col in dataset_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test")
        test_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))

        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if not col in dataset_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    def loss_fn(outputs: VQAOutput, labels: torch.Tensor, num_items_in_batch):
        func = torch.nn.CrossEntropyLoss()

        loss = func(outputs.logits, labels)

        return loss

    def metric_fn(predictions: EvalPrediction):
        logits: np.ndarray = predictions.predictions[0]
        labels, types = predictions.label_ids

        pred_labels = np.argmax(logits, axis=1)

        overall_accuracy = np.mean(pred_labels == labels)
        closed_accuracy = (
            np.mean(pred_labels[types == 0] == labels[types == 0])
            if np.any(types == 0)
            else 0
        )
        open_accuracy = (
            np.mean(pred_labels[types == 1] == labels[types == 1])
            if np.any(types == 1)
            else 0
        )

        return {
            "overall_accuracy": overall_accuracy,
            "closed_accuracy": closed_accuracy,
            "open_accuracy": open_accuracy,
        }

    # 8. Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_loss_func=loss_fn,
        compute_metrics=metric_fn,
        optimizers=(
            get_optimizer(
                model,
                learning_rate=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                max_steps=len(train_dataset) * training_args.num_train_epochs,
                warmup_ratio=training_args.warmup_ratio,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
            )
            if training_args.do_train
            else (None, None)
        ),
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        vqa_model: VQAModel = trainer.accelerator.unwrap_model(trainer.model.vqa_model)
        vqa_model.save_pretrained(os.path.join(training_args.output_dir, "vqa_model"))

        if model_args.model_type == "hf":
            if not model_args.freeze_vision_model:
                model: PreTrainedModel = trainer.accelerator.unwrap_model(
                    trainer.model.model
                )
                pth_path = os.path.join(training_args.output_dir, "model")

                # vision_model.save_pretrained(vision_path)
                # image_processor.save_pretrained(vision_path)

                os.makedirs(pth_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(pth_path, "pytorch_model.bin"))

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Test
    if training_args.do_predict:
        metrics = trainer.predict(test_dataset).metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
