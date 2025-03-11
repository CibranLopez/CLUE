import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np
import torch
import json
import os

import libraries.graph as clg

from torch_geometric.data import Data

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()

def generate_dataset(
        data_path,
        targets,
        data_folder
):
    """Generates a dataset from the raw data and saves it to disk.

    Args:
        data_path   (str): Path to the raw data.
        targets     (list): List of targets to be predicted.
        data_folder (str): Path to the folder where the dataset will be saved.

    Returns:
        None
    """
    # Define basic dataset parameters for tracking data
    dataset_parameters = {
        'input_folder':  data_path,
        'output_folder': data_folder,
        'target':        targets
    }
    
    if not os.path.exists(data_folder):
        os.system(f'mkdir {data_folder}')
    
    # Dump the dictionary with numpy arrays to a JSON file
    with open(f'{data_folder}/dataset_parameters.json', 'w') as json_file:
        json.dump(dataset_parameters, json_file)

    # Generate the raw dataset from scratch, and standardize it
    
    # Read all materials within the database
    dataset = []
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
                    nodes, edges, attributes = clg.graph_POSCAR_encoding(path_to_POSCAR)
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
                             y=torch.tensor(extracted_target, dtype=torch.float),
                             label=f'{material} {polymorph}'
                            )
    
                # Append to dataset and labels
                dataset.append(graph)

    
    torch.save(dataset, f'{data_folder}/dataset.pt')


def standardize_dataset(
        dataset,
        transformation=None
):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Args:
        dataset        (list): List containing graph structures.
        transformation (str):  Type of transformation strategy for edge attributes (None, 'inverse-quadratic').

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - labels_std         (list): Labels from valid graphs.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset and labels
    dataset_std = []
    for graph in dataset:
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())

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
        'target_mean':    np.array(target_mean.cpu().numpy()),
        'feat_mean':      np.array(feat_mean.cpu().numpy()),
        'edge_mean':      edge_mean.cpu().numpy(),
        'target_std':     np.array(target_std.cpu().numpy()),
        'feat_std':       np.array(feat_std.cpu().numpy()),
        'edge_std':       edge_std.cpu().numpy(),
        'scale':          scale.cpu().numpy()
    }
    return dataset_std, dataset_parameters


def standardize_dataset_from_keys(
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
    edge_mean   = standardized_parameters['edge_mean']
    feat_mean   = standardized_parameters['feat_mean']
    target_mean = standardized_parameters['target_mean']
    scale       = standardized_parameters['scale']
    edge_std    = standardized_parameters['edge_std']
    feat_std    = standardized_parameters['feat_std']
    target_std  = standardized_parameters['target_std']

    # Number of features per node
    n_features = dataset[0].num_node_features
    
    # Number of features per graph
    n_y = dataset[0].y.shape[0]
    
    # Check if non-linear standardization
    if standardized_parameters['transformation'] == 'inverse-quadratic':
        for data in dataset:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    for data in dataset:
        data.edge_attr = (data.edge_attr - edge_mean) * scale / edge_std

    for target_index in range(n_y):
        for data in dataset:
            data.y[target_index] = (data.y[target_index] - target_mean[target_index]) * scale / target_std[target_index]

    for feat_index in range(n_features):
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


def split_dataset(
        train_ratio,
        test_ratio,
        dataset
):
    """Splits the dataset into training, validation, and testing datasets.

    Args:
        train_ratio (float): Ratio of the dataset to be used for training.
        test_ratio  (float): Ratio of the dataset to be used for testing.
        dataset     (list):  List of graphs in PyTorch Geometric's Data format.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets.
    """
    # Define the sizes of the train, validation and test sets
    # Corresponds to the size wrt the number of unique materials in the dataset
    train_size = int(train_ratio * len(dataset))
    test_size  = int(test_ratio  * len(dataset))

    np.random.shuffle(dataset)

    # Random, fast splitting
    train_dataset = dataset[:train_size]
    val_dataset   = dataset[train_size:-test_size]
    test_dataset  = dataset[-test_size:]

    print(f'Number of training   graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of testing    graphs: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset


def load_datasets(
        files_names
):
    """Loads the training, validation, and testing datasets from disk.

    Args:
        files_names (dict): Dictionary containing the file names for the training, validation, and testing datasets.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets
    """
    train_dataset = torch.load(files_names['train_dt_std_name'], weights_only=False)
    val_dataset   = torch.load(files_names['val_dt_std_name'],   weights_only=False)
    test_dataset  = torch.load(files_names['test_dt_std_name'],  weights_only=False)

    standardized_parameters = load_json(files_names['std_param_name'])
    return train_dataset, val_dataset, test_dataset, standardized_parameters


def save_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        files_names
):
    """Saves the training, validation, and testing datasets to disk.

    Args:
        train_dataset (list): List of graphs in PyTorch Geometric's Data format.

    Returns:
        None
    """
    torch.save(train_dataset, files_names['train_dt_std_name'])
    torch.save(val_dataset,   files_names['val_dt_std_name'])
    torch.save(test_dataset,  files_names['test_dt_std_name'])


def save_json(
        file,
        file_name
):
    """Saves a dictionary to a JSON file.

    Args:
        file      (dict): Dictionary containing the data to be saved.
        file_name (str):  Path to the JSON file.

    Returns:
        None
    """
    # Convert torch tensors to numpy arrays
    for key, value in file.items():
        try:
            file[key] = value.tolist()
        except:
            pass

    # Dump the dictionary with numpy arrays to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(file, json_file)


def load_json(
        file_name
):
    """Loads a JSON file and converts torch tensors to torch tensors.

    Args:
        file_name (str): Path to the JSON file.
        to        (str): Convert torch tensors to torch tensors.

    Returns:
        dict: Dictionary containing the data from the JSON file.
    """
    # Load the data from the JSON file
    with open(file_name, 'r') as json_file:
        file = json.load(json_file)

    for key, value in file.items():
        try:
            file[key] = np.array(value, dtype=np.float32)
        except:
            pass
    return file


def parity_plot(
        train=np.array([np.nan, np.nan]),
        validation=np.array([np.nan, np.nan]),
        test=np.array([np.nan, np.nan]),
        figsize=(3, 3),
        save_to=None
):
    """Plots the computed vs. predicted values for the training, validation, and testing datasets.

    Args:
        train       (list): List containing the computed and predicted values for the training dataset.
        validation  (list): List containing the computed and predicted values for the validation dataset.
        test        (list): List containing the computed and predicted values for the testing dataset.
        figsize    (tuple): Size of the figure.

    Returns:
        None
    """
    x_train, y_train = train
    x_val,   y_val   = validation
    x_test,  y_test  = test

    plt.figure(figsize=figsize)

    if np.any(~np.isnan(train)):
        plt.plot(x_train, y_train, '.', label='Train')
    if np.any(~np.isnan(validation)):
        plt.plot(x_val, y_val, '.', label='Validation')
    if np.any(~np.isnan(test)):
        plt.plot(x_test, y_test, '.', label='Test')

    _min_, _max_ = get_min_max(train.flatten(), validation.flatten(), test.flatten())
    plt.xlabel('Computed')
    plt.ylabel('Predicted ')
    plt.plot([_min_, _max_], [_min_, _max_], '-r')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()


def losses_plot(
        train_losses,
        val_losses,
        to_log=True,
        figsize=(3, 3),
        save_to=None
):
    """Plots the training and validation losses.

    Args:
        train_losses (list): List containing the training losses.
        val_losses   (list): List containing the validation losses.
        to_log       (bool): If True, the losses are plotted in log scale.
        figsize      (tuple): Size of the figure.

    Returns:
        None
    """
    if to_log:
        plt.plot(np.log10(train_losses), label='Train loss')
        plt.plot(np.log10(val_losses) , label='Val  loss')
    else:
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses,   label='Val  loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()


def get_min_max(*data):
    """Determine the minimum and maximum values in a stack of data.

    Args:
        data: list of torch.tensor

    Returns:
        _min_: float
        _max_: float
    """
    stack = np.concatenate(data)
    _min_ = np.nanmin(stack)
    _max_ = np.nanmax(stack)
    return _min_, _max_