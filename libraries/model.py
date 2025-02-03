import numpy               as np
import torch.nn.functional as F
import torch

from scipy.interpolate      import RBFInterpolator
from scipy.spatial          import ConvexHull, Delaunay
from torch_geometric.data   import Batch
from torch_geometric.loader import DataLoader
from torch.nn               import Linear
from torch_geometric.nn     import GraphConv, global_mean_pool
from sklearn.decomposition import PCA

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def estimate_uncertainty(
        r_dataset,
        r_labels,
        t_dataset,
        model,
        r_uncertainty_data
):

    # Create a DataLoader for the reference dataset
    r_embeddings = extract_embeddings(r_dataset, model)

    # Create a DataLoader for the target dataset
    t_embeddings = extract_embeddings(t_dataset, model)

    # Determine which points are in the interpolation/extrapolation regime
    #are_interpolated = is_interpolating(r_embeddings, t_embeddings)
    #are_interpolated = [0]
    
    pca = PCA(n_components=5)  # Reduce to 5 dimensions
    r_embeddings_reduced = pca.fit_transform(r_embeddings)
    hull = Delaunay(r_embeddings_reduced)

    t_embeddings_reduced = pca.transform(t_embeddings)
    
    tolerance = 1e-9
    are_interpolated = hull.find_simplex(t_embeddings_reduced) >= tolerance

    # Extract uncertainty of each reference example
    r_uncertainties = [r_uncertainty_data[label] for label in r_labels]
    
    # Create an interpolator
    interpolator = RBFInterpolator(r_embeddings, r_uncertainties, smoothing=0)

    # Look for the uncertainty of the target dataset
    prediction_uncertainty = interpolator(t_embeddings)
    return prediction_uncertainty, are_interpolated


def extract_embeddings(
        dataset,
        model
):
    """Extract embeddings from a dataset using a trained model.
    """
    # Create a DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        embedding = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            return_graph_embedding=True
        ).cpu().numpy()
        embeddings.append(embedding)

    # Concatenate all batch embeddings into a single array
    return np.concatenate(embeddings, axis=0)


def is_interpolating(
    r_embeddings,
    t_embeddings,
    tolerance=1e-9
):
    
    # Generate convex-hull structure
    hull = ConvexHull(r_embeddings)
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]

    # Return which datapoints are or not within the convex-hull
    return np.all(np.dot(t_embeddings, A.T) + b <= tolerance, axis=1)


def estimate_out_of_distribution(
        r_dataset,
        t_dataset,
        model
):
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
    return closest_distances.cpu().numpy()


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
        self.conv1 = GraphConv(features_channels, 32)
        self.conv2 = GraphConv(32, 32)
        
        # Define linear layers
        self.linconv1 = Linear(32, 32)
        self.linconv2 = Linear(32, 16)
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

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        if return_graph_embedding:
            return x
        x = x.relu()
        
        ## REGRESSION
        
        # Apply final linear layer to make prediction
        x = self.lin(x)
        return x

        
def train(
        model,
        criterion,
        train_loader,
        target_factor,
        target_mean,
        optimizer
):
    """Train the model using the provided optimizer and criterion on the training dataset.

    Args:
        model        (torch.nn.Module):             The model to train.
        optimizer    (torch.optim.Optimizer):       The optimizer to use for updating model parameters.
        criterion    (torch.nn.Module):             The loss function to use.
        train_loader (torch.utils.data.DataLoader): The training dataset loader.

    Returns:
        float: The average training loss.
    """
    
    model.train()
    train_loss = 0
    all_predictions   = []
    all_ground_truths = []
    for data in train_loader:  # Iterate in batches over the training dataset
        # Moving data to device
        data = data.to(device)
        
        # Perform a single forward pass
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
        
        # Compute the loss
        loss = criterion(out, data.y)
        
        # Accumulate the training loss
        train_loss += loss.item()

        # Append predictions and ground truths to lists
        all_predictions.append(out.detach())
        all_ground_truths.append(data.y.detach())
        
        # Derive gradients
        loss.backward()
        
        # Update parameters based on gradients
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
    
    # Compute the average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # Concatenate predictions and ground truths into single arrays
    all_predictions   = torch.cat(all_predictions) * target_factor + target_mean
    all_ground_truths = torch.cat(all_ground_truths) * target_factor + target_mean
    return avg_train_loss, all_predictions.cpu().numpy(), all_ground_truths.cpu().numpy()

def test(
        model,
        criterion,
        test_loader,
        target_factor,
        target_mean
):
    """Evaluate the performance of a given model on a test dataset.

    Args:
        model       (torch.nn.Module):             The model to evaluate.
        criterion   (torch.nn.Module):             The loss function to use.
        test_loader (torch.utils.data.DataLoader): The test dataset loader.

    Returns:
        float: The average loss on the test dataset.
    """
    
    model.eval()
    test_loss = 0
    all_predictions   = []
    all_ground_truths = []
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the train/test dataset
            # Moving data to device
            data = data.to(device)
            
            # Perform a single forward pass
            out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
            
            # Compute the loss
            loss = criterion(out, data.y)
            
            # Accumulate the training loss
            test_loss += loss.item()

            # Append predictions and ground truths to lists
            all_predictions.append(out.detach())
            all_ground_truths.append(data.y.detach())
    
    # Compute the average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Concatenate predictions and ground truths into single arrays
    all_predictions   = torch.cat(all_predictions) * target_factor + target_mean
    all_ground_truths = torch.cat(all_ground_truths) * target_factor + target_mean
    return avg_test_loss, all_predictions.cpu().numpy(), all_ground_truths.cpu().numpy()


def make_predictions(
        reference_dataset,
        reference_labels,
        pred_dataset,
        model,
        standardized_parameters,
        reference_uncertainty_data
):
    """Make predictions.

    Args:
        reference_dataset       (list):            The reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        pred_dataset            (list):            List of graphs in PyTorch Geometric's Data format for predictions.
        model                   (torch.nn.Module): PyTorch model for predictions.
        standardized_parameters (dict):            Parameters needed to re-scale predicted properties from the dataset.
        net_uncertainty         (float):           Uncertainty associated to the network (e.g., from k-fold validation).

    Returns:
        numpy.ndarray: Predicted values.
    """

    # Read dataset parameters for re-scaling
    target_mean = standardized_parameters['target_mean'].item()
    scale       = standardized_parameters['scale'].item()
    target_std  = standardized_parameters['target_std'].item()

    # Computing the predictions
    dataset = DataLoader(pred_dataset, batch_size=64, shuffle=False)

    predictions   = []
    uncertainties = []
    are_interp    = []
    with torch.no_grad():  # No gradients for prediction
        for data in dataset:
            # Moving data to device
            data = data.to(device)

            # Perform a single forward pass
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()

            # Estimate uncertainty
            uncer, interp = estimate_uncertainty(reference_dataset, reference_labels,
                                                 data.to_data_list(), model, reference_uncertainty_data)

            # Append predictions to lists
            predictions.append(pred.cpu().detach())
            uncertainties.append(uncer)
            are_interp.append(interp)

    # Concatenate predictions and ground truths into single arrays
    predictions   = np.array((torch.cat(predictions) * target_std / scale + target_mean).cpu().numpy())
    uncertainties = np.concatenate(uncertainties)
    are_interp    = np.concatenate(are_interp)
    return predictions, uncertainties, are_interp


class EarlyStopping():
    def __init__(self, patience=5, delta=0, model_name='model.pt'):
        """Initializes the EarlyStopping object. Saves a model if accuracy is improved.
        Declares early_stop = True if training does not improve in patience steps within a delta threshold.

        Args:
            patience   (int):   Number of steps with no improvement.
            delta      (float): Threshold for a score to be considered an improvement.
            model_name (str):   Name of the saved model checkpoint file.
        """
        self.patience = patience  # Number of steps with no improvement
        self.delta = delta  # Threshold for a score to be an improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_name = model_name

    def __call__(self, val_loss, model):
        """Call method to check and update early stopping.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model checkpoint if the validation loss has decreased.
        It uses model.module, allowing models loaded to nn.DataParallel.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if val_loss < self.val_loss_min:
            torch.save(model.module.state_dict(), self.model_name)
            self.val_loss_min = val_loss

