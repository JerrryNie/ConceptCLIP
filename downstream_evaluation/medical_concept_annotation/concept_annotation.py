import sys
import os

from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier
import torch
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import torchvision.transforms as transforms

# Initialize model and processor
model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
processor = transforms.Compose([
            transforms.Resize(size=(384, 384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
model = model.cuda()
model = model.eval()
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
preprocess = processor
device = torch.device('cuda')


# load data
data_dir =  "data/"
image_dir = os.path.join(data_dir, "images")
meta_dir = os.path.join(data_dir, "meta")
df_meta_test = pd.read_csv(os.path.join(meta_dir, "meta_test.csv"))  # concept label: 0 / 1
df = df_meta_test

# concepts and prompt templates
concepts = ["pigment network", "streaks", "pigmentation", "regression structures", "dots and globules", "blue whitish veil", "vascular structures"]

template1 = {
    "postive_template": "This is dermatoscopy of {}.",
    "negative_template": "This is dermatoscopy."
}

template2 = {
    "postive_template": "This is dermoscopy of {}.",
    "negative_template": "This is dermoscopy."
}

templates = [template1, template2]

alpha = 0.5

# eval the performance of zero-shot medical concept annotation
for template in templates:

    num_samples = df.shape[0]
    correct = 0
    all_logits = []
    all_label = []

    for concept in tqdm(concepts):

        pos_prompt = template["postive_template"].format(concept)
        neg_prompt = template["negative_template"].format(concept)
        
        prompts = [neg_prompt, pos_prompt]
        texts = tokenizer(prompts, context_length=77).to(device)

        for idx, row in tqdm(df.iterrows()):
            img_path = os.path.join(image_dir, row["derm"])
            img = Image.open(img_path).convert('RGB')
            img = preprocess(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            concept_name = concept.replace(" ", "_")
            label = row[concept_name]

            res = model(img, texts)

            image_features = res['image_features']  
            image_features = F.normalize(image_features, dim=-1)
            text_features = res['text_features']  
            text_features = F.normalize(text_features, dim=-1)

            logit_scale = res['logit_scale'] 
            logits = (logit_scale * image_features @ text_features.t()) 
            logits = torch.softmax(logits, dim=-1)

            local_logit_scale = res['concept_logit_scale']
            image_features_local = res['image_token_features']
            text_features_local = res['text_token_features'] 
            
            text_features_local_neg = text_features_local[0][-1, :] 
            text_features_local_pos = text_features_local[1][-1, :] 

            text_features_local = torch.stack([text_features_local_neg, text_features_local_pos])

            image_features_local = F.normalize(image_features_local, dim=-1)
            text_features_local = F.normalize(text_features_local, dim=-1)
                
            temp_logits_local = local_logit_scale * image_features_local @ text_features_local.t() 
            logits_local = temp_logits_local.mean(dim=1)
            logits_local = torch.softmax(logits_local, dim=-1)

            logits = alpha * logits + (1-alpha) * logits_local 
            logits = logits.cpu().detach().numpy()

            prediction = 1 if logits[0][1] > logits[0][0] else 0
            if prediction == label:
                correct += 1
            
            if label == 1:
                label = [0,1]
            else:
                label = [1,0]
            all_label.append(label)
            logits = torch.tensor(logits)
            logits = torch.softmax(logits, 1)

            all_logits.append(logits)

    acc = correct / (num_samples * len(concepts))
    for i in range(len(all_logits)):
        all_logits[i] = all_logits[i][0]
    
    auc = roc_auc_score(all_label, all_logits)

    print("template: ", template)
    print('AUC: %.4f' % auc)
