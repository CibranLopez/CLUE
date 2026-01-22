import numpy as np
import torch

import libraries.dataset as cld

target_folder = 'models/pymatgen-graph-generation/Fv-previous-model'

dataset = torch.load(f'{target_folder}/dataset.pt', weights_only=False)

material_labels = [data.label for data in dataset]

train_labels = np.genfromtxt(f'{target_folder}/train_labels.txt', dtype='str').tolist()
val_labels = np.genfromtxt(f'{target_folder}/validation_labels.txt', dtype='str').tolist()
test_labels = np.genfromtxt(f'{target_folder}/test_labels.txt', dtype='str').tolist()
print(len(train_labels), len(val_labels), len(test_labels))

def get_datasets(
        subset_labels,
        dataset_labels,
        dataset
):
    subset_labels  = np.array(subset_labels)
    dataset_labels = np.array(dataset_labels)
    
    dataset_idxs = []
    for dataset_idx, dataset_label in enumerate(dataset_labels):
        for subset_idx, subset_label in enumerate(subset_labels):
            if dataset_label.split()[0] == subset_label:
                dataset_idxs.append(dataset_idx)
        if not len(subset_labels):
            break
    return [dataset[idx] for idx in dataset_idxs]

train_dataset = get_datasets(train_labels, material_labels, dataset)
val_dataset   = get_datasets(val_labels,   material_labels, dataset)
test_dataset  = get_datasets(test_labels,  material_labels, dataset)
print(len(train_dataset), len(val_dataset), len(test_dataset))

torch.save(train_dataset, f'{target_folder}/train_dataset.pt')
torch.save(val_dataset, f'{target_folder}/val_dataset.pt')
torch.save(test_dataset, f'{target_folder}/test_dataset.pt')

standardized_parameters = cld.load_json(f'{target_folder}/standardized_parameters.json')
train_dataset_std = cld.standardize_dataset_from_keys(train_dataset, standardized_parameters)
#train_dataset_std, standardized_parameters = cld.standardize_dataset(train_dataset)
val_dataset_std  = cld.standardize_dataset_from_keys(val_dataset,  standardized_parameters)
test_dataset_std = cld.standardize_dataset_from_keys(test_dataset, standardized_parameters)

torch.save(train_dataset_std, f'{target_folder}/train_dataset_std.pt')
torch.save(val_dataset_std, f'{target_folder}/val_dataset_std.pt')
torch.save(test_dataset_std, f'{target_folder}/test_dataset_std.pt')
cld.save_json(standardized_parameters, f'{target_folder}/standardized_parameters.json')
