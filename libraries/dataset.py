import numpy as np
import torch
import json
import os

from libraries.graph      import graph_POSCAR_encoding
from torch_geometric.data import Data
from pymatgen.core        import Structure

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_dataset(
        data_path,
        targets,
        data_folder
):
    
    # Define basic dataset parameters for tracking data
    dataset_parameters = {
        'input_folder':       data_path,
        'output_folder':      data_folder,
        'target':             targets
    }
    
    if not os.path.exists(data_folder):
        os.system(f'mkdir {data_folder}')
    
    # Dump the dictionary with numpy arrays to a JSON file
    with open(f'{data_folder}/dataset_parameters.json', 'w') as json_file:
        json.dump(dataset_parameters, json_file)

    # Generate the raw dataset from scratch, and standardize it
    
    # Read all materials within the database
    dataset = []
    labels  = []
    for material in os.listdir(data_path):
        # Check polymorph is a folder
        path_to_material = f'{data_path}/{material}'
        if not os.path.isdir(path_to_material):
            continue
        
        print(material)
        for polymorph in os.listdir(path_to_material):
            # Path to folder containing the POSCAR
            path_to_POSCAR = f'{data_path}/{material}/{polymorph}'
            
            # Check that the folder is valid
            if os.path.exists(path_to_POSCAR):
                print(f'\t{polymorph}')
                
                try:
                    nodes, edges, attributes = graph_POSCAR_encoding(f'{path_to_POSCAR}/POSCAR')
                except:
                    print(f'\tError: {material} {polymorph} not loaded')
                    continue
    
                extracted_target = []
                for target in targets:
                    if target == 'EPA':  # Load ground state energy per atom
                        extracted_target.append(float(np.loadtxt(f'{path_to_POSCAR}/EPA')))
                    elif target == 'bandgap':  # Load band-gap
                        extracted_target.append(float(np.loadtxt(f'{path_to_POSCAR}/bandgap')))
                
                # Construct temporal graph structure
                graph = Data(x=nodes,
                             edge_index=edges.t().contiguous(),
                             edge_attr=attributes.ravel(),
                             y=torch.tensor(extracted_target, dtype=torch.float)
                            )
    
                # Append to dataset and labels
                dataset.append(graph)
                labels.append(f'{material}-{polymorph}')
    
    
    torch.save(labels,  f'{data_folder}/labels.pt')
    torch.save(dataset, f'{data_folder}/dataset.pt')
    
    # Standardize dataset
    dataset_std, labels_std, dataset_parameters = standardize_dataset(dataset, labels,
                                                                      transformation='inverse-quadratic')

    torch.save(dataset_std, f'{data_folder}/standardized_dataset.pt')
    torch.save(labels_std,  f'{data_folder}/standardized_labels.pt')
    
    # Convert torch tensors to numpy arrays
    numpy_dict = {}
    for key, value in dataset_parameters.items():
        try:
            numpy_dict[key] = value.cpu().numpy().tolist()
        except:
            numpy_dict[key] = value
    
    # Dump the dictionary with numpy arrays to a JSON file
    with open(f'{data_folder}/standardized_parameters.json', 'w') as json_file:
        json.dump(numpy_dict, json_file)


def standardize_dataset(
        dataset,
        labels,
        transformation=None
):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Args:
        dataset        (list): List containing graph structures.
        labels         (list): List containing graph labels.
        transformation (str):  Type of transformation strategy for edge attributes (None, 'inverse-quadratic').

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - labels_std         (list): Labels from valid graphs.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset and labels
    dataset_std = []
    labels_std  = []
    for graph, label in zip(dataset, labels):
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())
            labels_std.append(label)

    # Number of graphs
    n_graphs = len(dataset_std)
    
    # Number of features per node
    n_features = dataset_std[0].num_node_features
    
    # Number of features per graph
    n_y = dataset_std[0].y.shape[0]
    
    # Check if non-linear standardization
    if transformation == 'inverse-quadratic':
        for data in dataset_std:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    # Compute means
    target_mean = torch.zeros(n_y)
    for target_index in range(n_y):
        target_mean[target_index] = sum([data.y[target_index] for data in dataset_std]) / n_graphs
    
    edge_mean = sum([data.edge_attr.mean() for data in dataset_std]) / n_graphs
    
    # Compute standard deviations
    target_std = torch.zeros(n_y)
    for target_index in range(n_y):
        target_std[target_index] = torch.sqrt(sum([(data.y[target_index] - target_mean[target_index]).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    edge_std = torch.sqrt(sum([(data.edge_attr - edge_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    # In case we want to increase the values of the normalization
    scale = torch.tensor(1e0)

    target_factor = target_std / scale
    edge_factor   = edge_std   / scale

    # Update normalized values into the database
    for data in dataset_std:
        data.y         = (data.y         - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean)   / edge_factor

    # Same for the node features
    feat_mean = torch.zeros(n_features)
    feat_std  = torch.zeros(n_features)
    for feat_index in range(n_features):
        # Compute mean
        temp_feat_mean = sum([data.x[:, feat_index].mean() for data in dataset_std]) / n_graphs
        
        # Compute standard deviations
        temp_feat_std = torch.sqrt(sum([(data.x[:, feat_index] - temp_feat_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))

        # Update normalized values into the database
        for data in dataset_std:
            data.x[:, feat_index] = (data.x[:, feat_index] - temp_feat_mean) * scale / temp_feat_std
        
        # Append corresponing values for saving
        feat_mean[feat_index] = temp_feat_mean
        feat_std[feat_index]  = temp_feat_std

    # Create and save as a dictionary
    dataset_parameters = {
        'transformation': transformation,
        'target_mean':    target_mean,
        'feat_mean':      feat_mean,
        'edge_mean':      edge_mean,
        'target_std':     target_std,
        'edge_std':       edge_std,
        'feat_std':       feat_std,
        'scale':          scale
    }
    return dataset_std, labels_std, dataset_parameters


def standarize_dataset_from_keys(
        dataset,
        standardized_parameters
):
    """Standardize the dataset. Non-linear normalization is also implemented.

    Args:
        dataset                 (list):  List of graphs in PyTorch Geometric's Data format.
        standardized_parameters (dict):  Parameters needed to re-scale predicted properties from the dataset.

    Returns:
        list: Standardized dataset.
    """

    # Read dataset parameters for re-scaling
    edge_mean = standardized_parameters['edge_mean']
    feat_mean = standardized_parameters['feat_mean']
    scale     = standardized_parameters['scale']
    edge_std  = standardized_parameters['edge_std']
    feat_std  = standardized_parameters['feat_std']

    # Check if non-linear standardization
    if standardized_parameters['transformation'] == 'inverse-quadratic':
        for data in dataset:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    for data in dataset:
        data.edge_attr = (data.edge_attr - edge_mean) * scale / edge_std

    for feat_index in range(dataset[0].num_node_features):
        for data in dataset:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) * scale / feat_std[feat_index]
    return dataset


def check_finite_attributes(
        data
):
    """
    Checks if all node and edge attributes in the graph are finite (i.e., not NaN, inf, or -inf).

    Args:
        data: A graph object containing node attributes (`data.x`) and edge attributes (`data.edge_attr`).

    Returns:
        bool: 
            - True if all node and edge attributes are finite.
            - False if any node or edge attributes are NaN, inf, or -inf.
    """
    # Check node attributes
    if not torch.any(torch.isfinite(data.x)):
        return False

    # Check edge attributes
    if not torch.any(torch.isfinite(data.edge_attr)):
        return False
    return True