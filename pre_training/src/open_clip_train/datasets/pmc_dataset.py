import logging
import random
import pandas as pd
import jsonlines
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence

from .utils import csv_loader, jsonl_loader


class PmcDataset(Dataset):
    def __init__(self, args, input_filename, transforms, tokenizer, is_train):
        print(f'Loading jsonl data from {input_filename}.')

        self.args = args
        self.max_txt_len = tokenizer.context_length
        suffix = input_filename.split('.')[-1]
        loader = {
            'csv': csv_loader,
            'jsonl': jsonl_loader,
        }[suffix]
        self.images, self.captions, self.metas, self.meta_lens = loader(
            input_filename=input_filename,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            meta_key=args.meta_key,
            sep=args.csv_separator
        )
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.recur = False
        if hasattr(self.tokenizer, 'tokenizer'):
            self.tokenizer = self.tokenizer.tokenizer
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self.recur = True
        print(f'self.recur: {self.recur}')

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            output = dict.fromkeys(["image", "text_ids", "text_masks", "metas"], None)
            if self.args.image_dir is None:
                image_path = self.images[idx]
            else:
                image_path = f'{self.args.image_dir}/{self.images[idx]}'
            image = self.transforms(Image.open(str(image_path)).convert('RGB'))
            caption = str(self.captions[idx])
            if self.recur:
                text_tokens = self.tokenizer(caption, return_tensors='pt', max_length=self.max_txt_len, truncation=True)
                text_ids = text_tokens['input_ids'][0]
                text_masks = text_tokens['attention_mask'][0].bool()
            else:
                text_ids = self.tokenizer(caption)[0]
                text_masks = None

            output.update({
                "image": image,
                "text_ids": text_ids,
                "text_masks": text_masks
            })
            if self.metas:
                output["metas"] = torch.tensor(self.metas[idx], dtype=torch.long)
                output["span_nums"] = torch.tensor(self.meta_lens[idx], dtype=torch.long)
            return output
        # catch the image reading error
        except UnidentifiedImageError as e:
            logging.error(f'Error at index {idx}: {e}')
            return self.__getitem__(random.randint(0, len(self) - 1))

    def collate_fn(self, batch):
        out_dict = {}
        keys = batch[0].keys()
        for key in keys:
            if key == "image":
                out_dict[key] = torch.stack([item[key] for item in batch], dim=0)
            elif key == "text_ids":
                # need to pad different text sequences into the same length
                texts = [item[key] for item in batch]
                padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0)
                out_dict[key] = padded_texts
            elif key == "text_masks" and batch[0][key] is not None:
                # need to pad different text sequences into the same length
                masks = [item[key] for item in batch]
                padded_masks = pad_sequence(masks, batch_first=True, padding_value=False)
                out_dict[key] = padded_masks
            elif key == "metas":
                if batch[0][key] is not None:
                    metas = [item[key] for item in batch]
                    # "-1" is for the subsequent "+1" (in the forward operation of the main model) so that the padding value can be 0 and other all values can be added up by 1
                    padded_metas = pad_sequence(metas, batch_first=True, padding_value=-1)
                    out_dict[key] = padded_metas
                else:
                    out_dict[key] = None
                    out_dict["span_nums"] = None
                    out_dict["repeated_vector"] = None
            elif key == "span_nums":
                span_nums = torch.tensor([item[key] for item in batch], dtype=torch.long)
                out_dict[key] = span_nums
                repeated_vector = torch.repeat_interleave(torch.arange(len(span_nums), device=span_nums.device), span_nums)
                out_dict["repeated_vector"] = repeated_vector
            else: # vqa_answer, vqa_labels, vqa_scores, qid
                out_dict[key] = [item[key] for item in batch]
        return out_dict['image'], out_dict['text_ids'], out_dict['text_masks'], out_dict['metas'], out_dict['span_nums'], out_dict['repeated_vector']
