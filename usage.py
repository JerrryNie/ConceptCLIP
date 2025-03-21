from transformers import AutoModel, AutoConfig, AutoProcessor
import torch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from open_clip import get_tokenizer, build_zero_shot_classifier
from PIL import Image

model_name = 'JerrryNie/ConceptCLIP'
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

template = 'a medical image of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]
test_imgs = [
    'bone_X-ray.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
]
context_length = 77
dataset_dir = 'example_data/'

imgs = [Image.open(os.path.join(dataset_dir, img)).convert('RGB') for img in test_imgs]
inputs = processor(images=imgs, text=[template + l for l in labels], return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    res = model(**inputs)
    image_features = res['image_features']
    image_token_features = res['image_token_features']
    text_features = res['text_features']
    logit_scale = res['logit_scale']

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')

import matplotlib.pyplot as plt

def plot_images_with_metadata(images, metadata):
    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(5, 5 * num_images))

    for i, (img_path, metadata) in enumerate(zip(images, metadata)):
        img = Image.open(os.path.join(dataset_dir, img_path))
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{metadata['filename']}\n{metadata['top_probs']}", fontsize=14)

    plt.tight_layout()
    plt.show()

metadata_list = []

top_k = 3
for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]
    img_name = img.split('/')[-1]

    top_probs = []
    top_k = len(labels) if top_k == -1 else top_k
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        top_probs.append(f"{labels[jth_index]}: {logits[i][jth_index] * 100:.1f}")

    metadata = {'filename': img_name, 'top_probs': '\n'.join(top_probs)}
    metadata_list.append(metadata)

plot_images_with_metadata(test_imgs, metadata_list)
