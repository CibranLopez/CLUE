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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
    #r_embeddings[r_embeddings<0] *= 1e-8
    #t_embeddings[t_embeddings<0] *= 1e-8

    # Extract labels from r_dataset
    r_labels = [data.label for data in r_dataset]

    # Determine which points are in the interpolation/extrapolation regime
    #t_interpolations = is_interpolating(r_embeddings, t_embeddings, n_components=5)
    t_interpolations = knn_ood_score(r_embeddings, t_embeddings)

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
    interpolating_method='RBF',
    novelty_k=None
):
    """Estimate the uncertainty of the target dataset by interpolation,
    scaling it by (1 + novelty) where novelty is the average kNN distance.

    Sets the uncertainty of extrapolated points to the maximum uncertainty between the expected one and that
    of the closest point in the reference dataset. Uncertainties for interpolated points remain unchanged.

    Args:
        r_embeddings         (numpy.ndarray): Reference embeddings.
        r_labels             (list):          Reference labels.
        r_uncertainty_data   (dict):          Uncertainty data for the reference dataset.
        t_embeddings         (numpy.ndarray): Target embeddings.
        t_interpolations     (numpy.ndarray): Boolean array indicating if the target embeddings are interpolated.
        interpolating_method (str):           Interpolation method ('RBF' or 'spline').
        novelty_k            (int):           Number of neighbors for novelty estimation.

    Returns:
        numpy.ndarray: Uncertainties of the target dataset.
    """
    # Get adaptative k-NN in case it is not provided
    novelty_k = min(5, len(r_embeddings)//10) if novelty_k is None else novelty_k
    
    # Extract uncertainties for each reference example
    r_uncertainties = np.asarray([r_uncertainty_data[label] for label in r_labels])

    # Compute novelty using kNN distance
    nbrs = NearestNeighbors(n_neighbors=novelty_k, algorithm="auto").fit(r_embeddings)

    if interpolating_method in ['RBF', 'spline']:
        if interpolating_method == 'RBF':
            interpolator = RBFInterpolator(r_embeddings, r_uncertainties, smoothing=0)
        elif interpolating_method == 'spline':
            interpolator = CubicSpline(r_embeddings, r_uncertainties)

        # Interpolate uncertainties for the target dataset
        t_uncertainties = interpolator(t_embeddings)

    elif interpolating_method == 'kNN':
        # k-NN weighted uncertainty
        tgt_dists, tgt_indices = nbrs.kneighbors(t_embeddings)

        method = 'inverse'
        if   method == 'exponential': weights = np.exp(- np.power(tgt_dists / 2*tgt_dists.mean(), 2))
        elif method == 'lineal':      weights = 1 - tgt_dists / tgt_dists.max()
        elif method == 'inverse':     weights = 1 / tgt_dists

        # Normalize weights
        weights /= weights.sum(axis=1, keepdims=True)

        # Weighted mean of uncertainties
        t_uncertainties = np.sum(weights * r_uncertainties[tgt_indices], axis=1)

    else:
        raise ValueError(f"Unsupported interpolation method: {interpolating_method}")

    # Mean k-NN distances for reference set
    ref_knn_dists, _ = nbrs.kneighbors(r_embeddings)
    ref_knn_means = np.mean(ref_knn_dists, axis=1)
    
    # Normalization factor based on percentile of reference mean distances
    norm_factor = np.percentile(ref_knn_means, 99)

    # Mean k-NN distances for target set
    tgt_knn_dists, indices = nbrs.kneighbors(t_embeddings)
    tgt_knn_means = np.mean(tgt_knn_dists, axis=1)  # average distance to k nearest neighbors

    # Normalized novelty
    novelty = tgt_knn_means / norm_factor
    
    # Apply novelty scaling
    t_uncertainties *= (1.0 + novelty)
    t_uncertainties = np.abs(t_uncertainties)

    # First neighbors list
    closest_indices = indices[:, 0]

    # Update uncertainties for extrapolated points
    # it has to be that of the maximum when in absolute (but value not in absolute)
    extrapolated_mask = ~t_interpolations
    t_uncertainties[extrapolated_mask] = np.maximum(
        t_uncertainties[extrapolated_mask],
        np.abs(r_uncertainties[closest_indices[extrapolated_mask]])
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


def knn_ood_score(
    r_embeddings,
    t_embeddings,
    n_neighbors=5
):
    """
    Compute an OOD score for test points based on k-NN distance to training points.

    Parameters
    ----------
    train_embeddings : np.ndarray, shape (n_train, d)
        Latent representations of the training set.
    test_embeddings : np.ndarray, shape (n_test, d)
        Latent representations of the test set.
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    ood_scores : np.ndarray, shape (n_test,)
        Mean distance to k nearest neighbors in the training set.
        Larger values = more likely OOD.
    """
    # Fit nearest neighbors model on training embeddings
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(r_embeddings)
    
    # Find distances to the k nearest neighbors for each test point
    distances, _ = nbrs.kneighbors(t_embeddings)
    
    # Use mean distance as OOD score
    scores = distances.mean(axis=1)
    
    threshold = np.percentile(scores, 90)  # mark top 5% farthest points as OOD
    # which is in fact similar to comparing to the mean (Gaussian distribution)
    ood_flags = scores < threshold
    return ood_flags


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
        self.conv1 = GraphConv(features_channels, 32)
        self.conv2 = GraphConv(32, 32)

        # Define linear layers
        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, 8)
        self.lin  = Linear(8, 1)
        
        self.pdropout = pdropout

    def forward(
            self,
            batch,
            return_graph_embedding=False
    ):
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
        x = x.relu()
        x = self.lin2(x)
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
