import os
import numpy as np
import torch
from tqdm import tqdm


def get_features(model, cur_dataloader, ann_path, model_name, cache_dir='linear_probing_cache', overwrite=False):
    all_features = []
    all_labels = []
    for p in model.parameters():
        p.requires_grad = False

    ann_basename = os.path.basename(ann_path)
    _cache_dir = os.path.join(cache_dir, ann_basename, model_name)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)

    features_cache_path = os.path.join(_cache_dir, f'{ann_basename}_features.npy')
    labels_cache_path = os.path.join(_cache_dir, f'{ann_basename}_labels.npy')

    if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path) and not overwrite:
        print(f'Loading features from {features_cache_path} and labels from {labels_cache_path}')
        all_features = np.load(features_cache_path, allow_pickle=True)
        all_labels = np.load(labels_cache_path, allow_pickle=True)
        return all_features, all_labels
    else:
        print(f'Extracting features from {ann_path}')
        with torch.no_grad():
            for batch in tqdm(cur_dataloader):
                images, labels = batch
                images = images.cuda()
                labels = labels.cuda()  # Move labels to GPU if necessary

                output = model.encode_image(image=images)
                if isinstance(output, tuple):
                    output = output[0]
                features = output.detach().cpu().numpy()  # Detach and move to CPU as numpy

                all_features.append(features)
                all_labels.append(labels.cpu().numpy())  # Move labels to CPU as numpy

                # Delete GPU tensors to free memory
                del images, output, features
                torch.cuda.empty_cache()

        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)

        # Debugging: Print shapes and data types
        print(f'all_features shape: {all_features.shape}, dtype: {all_features.dtype}')
        print(f'all_labels shape: {all_labels.shape}, dtype: {all_labels.dtype}')

        # Save features and labels separately
        np.save(features_cache_path, all_features)
        np.save(labels_cache_path, all_labels)

    return all_features, all_labels

    return all_features, all_labels


# Function to shuffle and then sample a sequential subset of the dataset
def shuffle_and_sample_data(features, labels, percentage, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Shuffle the indices
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    # Shuffle features and labels according to the shuffled indices
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]

    # Calculate the number of samples to select
    num_samples = int(len(features) * percentage / 100)

    # Select the first `num_samples` from the shuffled dataset
    sampled_features = shuffled_features[:num_samples]
    sampled_labels = shuffled_labels[:num_samples]

    return sampled_features, sampled_labels