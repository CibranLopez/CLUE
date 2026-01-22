import numpy as np
import torch

import libraries.dataset as cld

target_folder = 'models/MP-Fv'

dataset = torch.load(f'{target_folder}/dataset.pt', weights_only=False)

labels = [data.label for data in dataset]
unique_labels = np.unique(labels)

np.random.shuffle(unique_labels)
n_unique_labels = len(unique_labels)

train_labels = unique_labels[:int(0.8*n_unique_labels)]
val_labels = unique_labels[int(0.8*n_unique_labels):int(0.9*n_unique_labels)]
test_labels = unique_labels[int(0.9*n_unique_labels):]
print(len(train_labels), len(val_labels), len(test_labels))

train_dataset = []
val_dataset = []
test_dataset = []
for data in dataset:
    if data.label in train_labels:
        train_dataset.append(data)
    elif data.label in val_labels:
        val_dataset.append(data)
    elif data.label in test_labels:
        test_dataset.append(data)
print(len(train_dataset), len(val_dataset), len(test_dataset))

torch.save(train_dataset, f'{target_folder}/train_dataset.pt')
torch.save(val_dataset, f'{target_folder}/val_dataset.pt')
torch.save(test_dataset, f'{target_folder}/test_dataset.pt')

#standardized_parameters = cld.load_json(f'{target_folder}/standardized_parameters.json')
#train_dataset_std = cld.standardize_dataset_from_keys(train_dataset, standardized_parameters)
train_dataset_std, standardized_parameters = cld.standardize_dataset(train_dataset)
val_dataset_std  = cld.standardize_dataset_from_keys(val_dataset,  standardized_parameters)
test_dataset_std = cld.standardize_dataset_from_keys(test_dataset, standardized_parameters)

torch.save(train_dataset_std, f'{target_folder}/train_dataset_std.pt')
torch.save(val_dataset_std, f'{target_folder}/val_dataset_std.pt')
torch.save(test_dataset_std, f'{target_folder}/test_dataset_std.pt')
cld.save_json(standardized_parameters, f'{target_folder}/standardized_parameters.json')
