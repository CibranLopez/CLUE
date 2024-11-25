import numpy               as np
import torch               as torch
import torch.nn.functional as F
import os

from scipy.optimize          import curve_fit
from torch_geometric.loader  import DataLoader
from torch_geometric.data    import Data, Batch
from torch.nn                import Linear
from torch_geometric.nn      import GraphConv, global_mean_pool
from pymatgen.core.structure import Structure

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNN(torch.nn.Module):
    """Graph convolution neural network.
    """
    
    def __init__(self, features_channels, pdropout):
        """Initializes the Graph Convolutional Neural Network.

        Args:
            features_channels (int):   Number of input features.
            pdropout          (float): Dropout probability for regularization.

        Returns:
            None
        """
        
        super(GCNN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(12345)
        
        # Define graph convolution layers
        self.conv1 = GraphConv(features_channels, 512)
        self.conv2 = GraphConv(512, 512)
        
        # Define linear layers
        self.linconv1 = Linear(512, 64)
        self.linconv2 = Linear(64, 16)
        self.lin      = Linear(16, 1)
        
        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr, batch, return_graph_embedding=False):
        ## CONVOLUTION
        
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()

        ## POOLING
        
        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        if return_graph_embedding:
            return x

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        x = x.relu()
        
        ## REGRESSION
        
        # Apply final linear layer to make prediction
        x = self.lin(x)
        return x

def estimate_out_of_distribution(r_dataset, t_dataset, model):
    """We use the pooling from a graph neural network, which is a vector representation of the
    material, to assess the similarity between the target graph with respect to the dataset.

    Args:
        r_dataset (list):            The reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        t_dataset (list):            Target dataset to assess the similarity on.
        model     (torch.nn.Module): PyTorch model for predictions.

    Returns:
        list ints:   Indexes of the closest example referred to the reference dataset.
        list floats: Distances to the distribution.
    """

    # Generate embeddings for target dataset
    t_batch = Batch.from_data_list(t_dataset).to(device)
    t_embeddings = model(t_batch.x, t_batch.edge_index, t_batch.edge_attr, t_batch.batch,
                         return_graph_embedding=True)

    closest_distances = torch.full((t_embeddings.size(0),), float('inf'), device=device)

    # Create a DataLoader for the reference dataset
    r_loader = DataLoader(r_dataset, batch_size=64, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    for r_batch in r_loader:
        r_batch = r_batch.to(device)
        r_embeddings = model(
            r_batch.x, r_batch.edge_index, r_batch.edge_attr, r_batch.batch,
            return_graph_embedding=True
        )

        # Compute pairwise distances
        pairwise_distances = torch.cdist(t_embeddings, r_embeddings)

        # Update global closest distances
        closest_distances = torch.minimum(closest_distances, torch.min(pairwise_distances, dim=1).values)

    # Move results to CPU and return as a list
    return np.sqrt(closest_distances.cpu().numpy())