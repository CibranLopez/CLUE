import numpy               as np
import torch.nn.functional as F
import torch.nn            as nn
import torch
import os

from scipy.interpolate      import RBFInterpolator, CubicSpline
from scipy.spatial          import Delaunay
from torch_geometric.data   import Batch
from torch_geometric.loader import DataLoader
from torch.nn               import Linear, BatchNorm1d
from torch_geometric.nn     import GraphConv, global_mean_pool
from sklearn.decomposition  import PCA

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_uncertainty(
        r_dataset,
        t_dataset,
        model,
        r_uncertainty_data
):
    """Estimate uncertainty on predictions and whether the target dataset is in the interpolation regime.

    Args:
        r_dataset          (list):            Reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        t_dataset          (list):            Target dataset, as a list of graphs in PyTorch Geometric's Data format.
        model              (torch.nn.Module): The trained model.
        r_uncertainty_data (dict):            Uncertainty data for the reference dataset.

    Returns:
        numpy.ndarray: Uncertainties of the target dataset.
        numpy.ndarray: Boolean array indicating if the target embeddings
    """

    # Create a DataLoader for the reference dataset
    r_embeddings = extract_embeddings(r_dataset, model)

    # Create a DataLoader for the target dataset
    t_embeddings = extract_embeddings(t_dataset, model)

    # It fails with illdefined spaces, thus with ReLU activation function as well
    r_embeddings[r_embeddings<0] *= 1e-8
    t_embeddings[t_embeddings<0] *= 1e-8

    # Extract labels from r_dataset
    r_labels = [data.label for data in r_dataset]

    # Determine which points are in the interpolation/extrapolation regime
    t_interpolations = is_interpolating(r_embeddings, t_embeddings)
    #t_interpolations = np.array([True] * len(t_embeddings))

    # Determine the uncertainty on the predictions
    t_uncertainties = estimate_uncertainty(r_embeddings, r_labels,
                                           r_uncertainty_data,
                                           t_embeddings,
                                           t_interpolations)

    return t_uncertainties, t_interpolations


def estimate_uncertainty(
    r_embeddings,
    r_labels,
    r_uncertainty_data,
    t_embeddings,
    t_interpolations,
    interpolating_method='RBF'
):
    """Estimate the uncertainty of the target dataset by interpolation.

    Sets the uncertainty of extrapolated points to the maximum uncertainty between the expected one and that
    of the closest point in the reference dataset. Uncertainties for interpolated points remain unchanged.

    Args:
        r_embeddings         (numpy.ndarray): Reference embeddings.
        r_labels             (list):          Reference labels.
        r_uncertainty_data   (dict):          Uncertainty data for the reference dataset.
        t_embeddings         (numpy.ndarray): Target embeddings.
        t_interpolations     (numpy.ndarray): Boolean array indicating if the target embeddings are interpolated.
        interpolating_method (str):           Interpolation method ('RBF' or 'spline').

    Returns:
        numpy.ndarray: Uncertainties of the target dataset.
    """
    # Extract uncertainties for each reference example
    r_uncertainties = np.asarray([r_uncertainty_data[label] for label in r_labels])

    # Select interpolation method
    if interpolating_method == 'RBF':
        interpolator = RBFInterpolator(r_embeddings, r_uncertainties, smoothing=0)
    elif interpolating_method == 'spline':
        interpolator = CubicSpline(r_embeddings, r_uncertainties)
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolating_method}")

    # Interpolate uncertainties for the target dataset
    t_uncertainties = interpolator(t_embeddings)

    # Compute pairwise distances between target and reference embeddings
    distances = np.linalg.norm(t_embeddings[:, None, :] - r_embeddings[None, :, :], axis=2)

    # Find the closest reference point for each target embedding
    closest_indices = np.argmin(distances, axis=1)

    # Update uncertainties for extrapolated points
    extrapolated_mask = ~t_interpolations
    t_uncertainties[extrapolated_mask] = np.maximum(
        t_uncertainties[extrapolated_mask],
        r_uncertainties[closest_indices[extrapolated_mask]]
    )

    return t_uncertainties


def extract_embeddings(
        dataset,
        model
):
    """Extract embeddings from a dataset using a trained model.

    Args:
        dataset (list):            Dataset, as a list of graphs in PyTorch Geometric's Data format.
        model   (torch.nn.Module): The trained model.

    Returns:
        numpy.ndarray: Embeddings extracted from the dataset.
    """
    # Create a DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        embedding = model(batch, return_graph_embedding=True).cpu().numpy()
        embeddings.append(embedding)

    # Concatenate all batch embeddings into a single array
    return np.concatenate(embeddings, axis=0)


def is_interpolating(
    r_embeddings,
    t_embeddings,
    n_components=None
):
    """Check if the target embeddings are in the interpolation regime. If n_components is not None,
    it reduces dimensionality of embeddings to n_components dimensions.

    Args:
        r_embeddings (numpy.ndarray): Reference embeddings.
        t_embeddings (numpy.ndarray): Target embeddings.
        n_components (int, None):     Number of components for PCA. If None, PCA is not performed.

    Returns:
        numpy.ndarray: Boolean array indicating if the target embeddings are interpolated.
    """
    if n_components is not None:
        pca = PCA(n_components=n_components)
        r_embeddings = pca.fit_transform(r_embeddings)
        t_embeddings = pca.transform(t_embeddings)

    # Generate convex hull with reduced data (using Delaunay approach)
    hull = Delaunay(r_embeddings)

    # Check if the points are inside the convex hull
    simplex_indices = hull.find_simplex(t_embeddings)

    # Convert to boolean: True for interpolation, False for extrapolation
    are_interpolated = simplex_indices != -1
    return are_interpolated


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
    t_embeddings = model(t_batch, return_graph_embedding=True).cpu().numpy()

    closest_distances = torch.full((t_embeddings.size(0),), float('inf'), device=device)

    # Create a DataLoader for the reference dataset
    r_loader = DataLoader(r_dataset, batch_size=64, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    for r_batch in r_loader:
        r_batch = r_batch.to(device)
        r_embeddings = model(r_batch, return_graph_embedding=True).cpu().numpy()

        # Compute pairwise distances
        pairwise_distances = torch.cdist(t_embeddings, r_embeddings)

        # Update global closest distances
        closest_distances = torch.minimum(closest_distances, torch.min(pairwise_distances, dim=1).values)

    # Move results to CPU and return as a list
    return closest_distances.cpu().numpy()


class GCNN(
    torch.nn.Module
):
    """Graph convolution neural network.
    """
    
    def __init__(
            self,
            features_channels,
            pdropout
    ):
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

    def forward(
            self,
            batch,
            return_graph_embedding=False
    ):
        ## CONVOLUTION
        
        # Apply graph convolution with ReLU activation function
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = x.relu()
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = x.relu()

        ## POOLING
        
        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch.batch)  # [batch_size, hidden_channels]
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


class GCNN2(
    torch.nn.Module
):
    """Graph convolution neural network.
    """
    
    def __init__(
            self,
            features_channels,
            pdropout
    ):
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
        self.conv1 = GraphConv(features_channels, 64)
        self.conv2 = GraphConv(64, 128)
        
        # Define linear layers
        self.lin1 = Linear(128, 6)
        self.lin  = Linear(6, 1)
        
        #self.bn1 = BatchNorm1d(128)
        
        self.pdropout = pdropout

    def forward(
            self,
            batch,
            return_graph_embedding=False
    ):
        """Forward pass of the Graph Convolutional Neural Network.

        Args:
            x                    (torch.Tensor): Node features.
            edge_index           (torch.Tensor): Edge indices.
            edge_attr            (torch.Tensor): Edge attributes.
            batch                (torch.Tensor): Batch indices.
            return_graph_embedding (bool):        Return graph embeddings.

        Returns:
            torch.Tensor: Predicted values.
        """
        # Apply graph convolution with ReLU activation function
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = x.relu()
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = x.relu()

        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch.batch)  # [batch_size, hidden_channels]

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.lin1(x)
        #x = self.bn1(x)
        #x = x.relu()
        #x = self.lin2(x)
        if return_graph_embedding:
            return x
        x = x.relu()
        
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
    predictions   = []
    ground_truths = []
    for data in train_loader:  # Iterate in batches over the training dataset
        # Moving data to device
        data = data.to(device)
        
        # Perform a single forward pass
        out = model(data).flatten()
        
        # Compute the loss
        loss = criterion(out, data.y)
        
        # Accumulate the training loss
        train_loss += loss.item()

        # Append predictions and ground truths to lists
        predictions.append(out.detach().cpu().numpy())
        ground_truths.append(data.y.detach().cpu().numpy())
        
        # Derive gradients
        loss.backward()
        
        # Update parameters based on gradients
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
    
    # Compute the average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # Concatenate predictions and ground truths into single arrays
    predictions   = np.concatenate(predictions)   * target_factor + target_mean
    ground_truths = np.concatenate(ground_truths) * target_factor + target_mean
    return avg_train_loss, predictions, ground_truths


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
    predictions   = []
    ground_truths = []
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the train/test dataset
            # Moving data to device
            data = data.to(device)
            
            # Perform a single forward pass
            out = model(data).flatten()
            
            # Compute the loss
            loss = criterion(out, data.y)
            
            # Accumulate the training loss
            test_loss += loss.item()

            # Append predictions and ground truths to lists
            predictions.append(out.detach().cpu().numpy())
            ground_truths.append(data.y.detach().cpu().numpy())
    
    # Compute the average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Concatenate predictions and ground truths into single arrays
    predictions   = np.concatenate(predictions)   * target_factor + target_mean
    ground_truths = np.concatenate(ground_truths) * target_factor + target_mean
    return avg_test_loss, predictions, ground_truths


def forward_predictions(
        reference_dataset,
        pred_dataset,
        model,
        standardized_parameters,
        reference_uncertainty_data
):
    """Make predictions on a dataset using a trained model.

    Args:
        reference_dataset (list):            Reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        pred_dataset      (list):            Prediction dataset, as a list of graphs in PyTorch Geometric's Data format.
        model             (torch.nn.Module): The trained model.
        standardized_parameters (dict):      Standardized parameters for rescaling the predictions.
        reference_uncertainty_data (dict):   Uncertainty data for the reference dataset.
    Returns:
        numpy.ndarray: Predicted values.
    """
    model.eval()
    
    # Read dataset parameters for re-scaling
    target_mean  = standardized_parameters['target_mean']
    target_std   = standardized_parameters['target_std']
    target_scale = standardized_parameters['scale']

    # Read uncertainty parameters for re-scaling
    uncert_mean  = reference_uncertainty_data['uncert_mean']
    uncert_std   = reference_uncertainty_data['uncert_std']
    uncert_scale = reference_uncertainty_data['uncert_scale']

    # Computing the predictions
    dataset = DataLoader(pred_dataset, batch_size=128, shuffle=False, pin_memory=True)

    predictions    = []
    uncertainties  = []
    interpolations = []
    with torch.no_grad():  # No gradients for prediction
        for data in dataset:
            # Moving data to device
            data = data.to(device)

            # Perform a single forward pass
            pred = model(data).flatten().detach().cpu().numpy()

            # Estimate uncertainty
            uncert, interp = analyze_uncertainty(reference_dataset,
                                                 data.to_data_list(), model, reference_uncertainty_data['uncertainty_values'])

            # Append predictions to lists
            predictions.append(pred)
            uncertainties.append(uncert)
            interpolations.append(interp)

    # Concatenate predictions and ground truths into single arrays
    predictions    = np.concatenate(predictions)   * target_std / target_scale + target_mean  # De-standardize predictions
    uncertainties  = np.concatenate(uncertainties) * uncert_std / uncert_scale + uncert_mean  # De-standardize predictions
    interpolations = np.concatenate(interpolations)
    return predictions, uncertainties, interpolations


class EarlyStopping():
    def __init__(
            self,
            patience=5,
            delta=0,
            model_name='model.pt'
    ):
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
        self.val_loss_min = np.inf
        self.model_name = model_name

    def __call__(
            self,
            val_loss,
            model
    ):
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

    def save_checkpoint(
            self,
            val_loss,
            model
    ):
        """Save the model checkpoint if the validation loss has decreased.
        It uses model.module, allowing models loaded to nn.DataParallel.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if val_loss < self.val_loss_min:
            torch.save(model.module.state_dict(), self.model_name)
            self.val_loss_min = val_loss
            self.val_loss_min = np.inf


def load_model(
        n_node_features,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='eval'
):
    # Load Graph Neural Network model
    model = GCNN(features_channels=n_node_features, pdropout=pdropout)

    # Moving model to device
    model = model.to(device)

    if model_name is not None and os.path.exists(model_name):
        # Load Graph Neural Network model
        model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))

    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()

    # Allow data parallelization among multi-GPU
    model = nn.DataParallel(model)
    return model
