"""
Generative Machine Learning Approaches to Optimization

This module provides reusable utilities for generative optimization including:
- Sampling utilities (Sobol, Latin Hypercube)
- Gaussian Mixture Model (GMM) fitting and selection
- Conditional Flow Matching for inverse design
- Cluster analysis and validation utilities

Authors: Victor Alves and John R. Kitchin
"""

import numpy as np
import warnings
from scipy.stats.qmc import Sobol, LatinHypercube, scale
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from gmr import GMM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Default device for PyTorch
device = torch.device('cpu')


# =============================================================================
# Sampling Utilities
# =============================================================================

def generate_samples(bounds, n_samples=None, method='sobol', seed=42):
    """
    Generate samples in a bounded region with consistent strategy.

    Parameters
    ----------
    bounds : array-like, shape (n_dims, 2)
        [[low1, high1], [low2, high2], ...]
    n_samples : int, optional
        Number of samples. Default: 2^ceil(log2(20 * n_dims))
    method : str
        'sobol', 'latin', or 'uniform'
    seed : int
        Random seed

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dims)
    """
    bounds = np.atleast_2d(bounds)
    n_dims = len(bounds)

    if n_samples is None:
        # Heuristic: exponential in dimension, but reasonable
        n_samples = 2 ** int(np.ceil(np.log2(20 * n_dims)))
        n_samples = min(n_samples, 4096)

    lower = bounds[:, 0]
    upper = bounds[:, 1]

    if method == 'sobol':
        sampler = Sobol(d=n_dims, scramble=True, seed=seed)
        # Sobol works best with power of 2
        n_samples = 2 ** int(np.ceil(np.log2(n_samples)))
        raw = sampler.random(n=n_samples)
    elif method == 'latin':
        sampler = LatinHypercube(d=n_dims, seed=seed)
        raw = sampler.random(n=n_samples)
    else:  # uniform
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0, 1, (n_samples, n_dims))

    return scale(raw, lower, upper)


def estimate_sample_size(n_dims, n_components_expected=5, samples_per_param=20):
    """
    Estimate required sample size for GMM fitting.

    A GMM with k components in d dimensions has approximately:
    k * (1 + d + d(d+1)/2) parameters

    Parameters
    ----------
    n_dims : int
        Number of dimensions in the data
    n_components_expected : int
        Expected number of GMM components
    samples_per_param : int
        Samples per parameter (rule of thumb: 10-30)

    Returns
    -------
    n_samples : int
        Recommended sample size (power of 2 for Sobol)
    """
    n_params_per_component = 1 + n_dims + n_dims * (n_dims + 1) // 2
    total_params = n_components_expected * n_params_per_component
    n_samples = samples_per_param * total_params

    # Round up to power of 2 for Sobol
    n_samples = 2 ** int(np.ceil(np.log2(n_samples)))

    return min(n_samples, 8192)


# =============================================================================
# GMM Utilities
# =============================================================================

def best_gmm(X, max_components=None, criterion='bic',
             reg_covar=1e-6, n_init=3, patience=10,
             verbose=False):
    """
    Find the best GMM using information criteria with early stopping.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data
    max_components : int, optional
        Maximum components to try. Default: min(50, n_samples // (5 * n_features))
    criterion : str
        'bic' or 'aic' for model selection
    reg_covar : float
        Regularization for covariance matrices (prevents singularity)
    n_init : int
        Number of initializations per component count
    patience : int
        Stop if no improvement for this many components
    verbose : bool
        Print progress

    Returns
    -------
    gmm : GMM
        Best model from gmr library
    info : dict
        Dictionary with 'scores', 'best_k', 'all_models'
    """
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape

    if max_components is None:
        max_components = min(50, max(2, n_samples // (5 * n_features)))

    scores = []
    models = []
    best_score = np.inf
    best_k = 1
    no_improvement = 0

    for k in range(1, max_components + 1):
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type='full',
                reg_covar=reg_covar,
                n_init=n_init,
                random_state=42
            ).fit(X)

            score = model.bic(X) if criterion == 'bic' else model.aic(X)

        except Exception as e:
            if verbose:
                print(f"k={k}: Failed - {e}")
            continue

        scores.append(score)

        gmm = GMM(
            n_components=model.n_components,
            priors=model.weights_,
            means=model.means_,
            covariances=model.covariances_
        )
        models.append(gmm)

        if score < best_score:
            best_score = score
            best_k = len(models)  # index in models list
            no_improvement = 0
        else:
            no_improvement += 1

        if verbose:
            marker = " *" if no_improvement == 0 else ""
            print(f"k={k}: {criterion.upper()}={score:.1f}{marker}")

        if no_improvement >= patience:
            if verbose:
                print(f"Early stopping at k={k}")
            break

    info = {
        'scores': scores,
        'best_k': best_k,
        'all_models': models,
        'criterion': criterion
    }

    return models[best_k - 1], info


def best_gmm_bic(X, maxN=None):
    """Legacy wrapper for best_gmm with BIC criterion."""
    gmm, info = best_gmm(X, max_components=maxN, criterion='bic')
    # Attach extra attributes for compatibility
    gmm.models = info['all_models']
    gmm.bics = info['scores']
    return gmm


# =============================================================================
# Cluster Analysis
# =============================================================================

def cluster_stats(X, eps=0.5, min_samples=5, return_stats=False):
    """
    Compute mean and stdev for clusters in sampled data.

    Parameters
    ----------
    X : array-like, shape (n_samples,) or (n_samples, n_features)
        Data to cluster
    eps : float
        DBSCAN neighborhood radius
    min_samples : int
        Minimum samples for core point
    return_stats : bool
        If True, return dictionary of statistics

    Returns
    -------
    stats : dict (if return_stats=True)
        {label: {'mean': ..., 'std': ..., 'n': ...}}
    """
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    stats = {}
    for L in sorted(set(labels)):
        mask = labels == L
        cluster_data = X[mask]

        if X.shape[1] == 1:
            mean = np.mean(cluster_data)
            std = np.std(cluster_data)
            print(f'{L:3d}: {mean: 8.4f} +/- {std:.4f} (n={np.sum(mask)})')
        else:
            mean = np.mean(cluster_data, axis=0)
            std = np.std(cluster_data, axis=0)
            print(f'{L:3d}: mean={mean}, std={std} (n={np.sum(mask)})')

        stats[L] = {'mean': mean, 'std': std, 'n': np.sum(mask)}

    if return_stats:
        return stats


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_gmm(gmm, X, input_cols, output_cols=None, plot=True):
    """
    Validate GMM fit with parity plot and metrics.

    Parameters
    ----------
    gmm : GMM
        Fitted model
    X : ndarray
        Data used for fitting
    input_cols : list of int
        Column indices to condition on
    output_cols : list of int, optional
        Column indices to predict. Default: all columns not in input_cols
    plot : bool
        Whether to show parity plot

    Returns
    -------
    metrics : dict
        R2, MAE, RMSE for each output column
    """
    import matplotlib.pyplot as plt

    if output_cols is None:
        output_cols = [i for i in range(X.shape[1]) if i not in input_cols]

    inputs = X[:, input_cols]
    true_outputs = X[:, output_cols]
    pred_outputs = gmm.predict(input_cols, inputs)

    metrics = {}

    if plot:
        n_outputs = len(output_cols)
        fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4))
        if n_outputs == 1:
            axes = [axes]

    for i, col in enumerate(output_cols):
        true = true_outputs[:, i]
        pred = pred_outputs[:, i]

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))

        metrics[f'col_{col}'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

        if plot:
            ax = axes[i]
            ax.scatter(true, pred, alpha=0.5, s=10)
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            ax.plot(lims, lims, 'k--', lw=1)
            ax.set_xlabel(f'True (col {col})')
            ax.set_ylabel(f'Predicted (col {col})')
            ax.set_title(f'R² = {r2:.4f}')
            ax.set_aspect('equal')

    if plot:
        plt.tight_layout()
        plt.show()

    return metrics


# =============================================================================
# Conditional Flow Matching
# =============================================================================

class VelocityNetwork(nn.Module):
    """Neural network that predicts velocity field for flow matching."""

    def __init__(self, x_dim, c_dim, hidden_dim=128, n_layers=3):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim

        # Input: x, t, c -> output: velocity (same dim as x)
        layers = [nn.Linear(x_dim + 1 + c_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, x_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c):
        """Predict velocity at (x, t) conditioned on c."""
        # Ensure t has correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        # Concatenate inputs
        inp = torch.cat([x, t, c], dim=-1)
        return self.net(inp)


class ConditionalFlowMatching:
    """Conditional Flow Matching for optimization problems.

    Given data (x, y) where x are inputs and y are outputs,
    learns to generate x conditioned on y.

    Parameters
    ----------
    x_dim : int
        Dimension of variables to generate
    c_dim : int
        Dimension of conditioning variables
    hidden_dim : int
        Hidden layer size
    n_layers : int
        Number of hidden layers
    sigma_min : float
        Minimum noise level for flow matching
    device : torch.device, optional
        Device to use for computation
    """

    def __init__(self, x_dim, c_dim, hidden_dim=128, n_layers=3, sigma_min=0.01,
                 device_override=None):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.sigma_min = sigma_min
        self.device = device_override if device_override is not None else device

        self.model = VelocityNetwork(x_dim, c_dim, hidden_dim, n_layers).to(self.device)
        self.optimizer = None

        # Store normalization parameters
        self.x_mean = None
        self.x_std = None
        self.c_mean = None
        self.c_std = None

        # Store training history
        self.history = {}

    def _normalize(self, x, c):
        """Normalize data to zero mean and unit variance."""
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        c_norm = (c - self.c_mean) / (self.c_std + 1e-8)
        return x_norm, c_norm

    def _unnormalize_x(self, x_norm):
        """Unnormalize x back to original scale."""
        return x_norm * (self.x_std + 1e-8) + self.x_mean

    def fit(self, x_data, c_data, epochs=1000, batch_size=64, lr=1e-3,
            verbose=True, eval_every=50, n_eval_samples=50):
        """Train the flow matching model.

        Parameters
        ----------
        x_data : ndarray, shape (n_samples, x_dim)
            The variables to generate (inputs in optimization context)
        c_data : ndarray, shape (n_samples, c_dim)
            The conditioning variables (outputs in optimization context)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        verbose : bool
            Whether to show progress bar
        eval_every : int
            Evaluate accuracy every N epochs
        n_eval_samples : int
            Number of samples to generate for evaluation
        """
        # Compute normalization parameters
        self.x_mean = torch.tensor(x_data.mean(axis=0), dtype=torch.float32, device=self.device)
        self.x_std = torch.tensor(x_data.std(axis=0), dtype=torch.float32, device=self.device)
        self.c_mean = torch.tensor(c_data.mean(axis=0), dtype=torch.float32, device=self.device)
        self.c_std = torch.tensor(c_data.std(axis=0), dtype=torch.float32, device=self.device)

        # Convert to tensors
        x = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        c = torch.tensor(c_data, dtype=torch.float32, device=self.device)

        # Normalize
        x, c = self._normalize(x, c)

        dataset = TensorDataset(x, c)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # History tracking
        losses = []
        eval_epochs = []
        eval_mae = []
        eval_r2 = []

        iterator = tqdm(range(epochs), disable=not verbose, desc="Training")

        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            for x_batch, c_batch in loader:
                self.optimizer.zero_grad()

                # Sample time uniformly
                t = torch.rand(x_batch.shape[0], 1, device=self.device)

                # Sample noise
                noise = torch.randn_like(x_batch)

                # Interpolate: x_t = (1-t) * noise + t * x
                # This is the "optimal transport" path
                x_t = (1 - t) * noise + t * x_batch

                # Target velocity: dx/dt = x - noise
                target_v = x_batch - noise

                # Predict velocity
                pred_v = self.model(x_t, t, c_batch)

                # MSE loss
                loss = ((pred_v - target_v) ** 2).mean()

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            # Evaluate accuracy periodically
            if (epoch + 1) % eval_every == 0 or epoch == 0:
                mae, r2 = self._evaluate_accuracy(x_data, c_data, n_eval_samples)
                eval_epochs.append(epoch + 1)
                eval_mae.append(mae)
                eval_r2.append(r2)

                if verbose:
                    iterator.set_postfix(loss=f"{avg_loss:.4f}", MAE=f"{mae:.4f}", R2=f"{r2:.4f}")
            elif verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix(loss=f"{avg_loss:.6f}")

        # Store history
        self.history = {
            'loss': losses,
            'eval_epochs': eval_epochs,
            'eval_mae': eval_mae,
            'eval_r2': eval_r2
        }

        return self.history

    def _evaluate_accuracy(self, x_data, c_data, n_samples=50):
        """Evaluate model accuracy on a subset of the data."""
        self.model.eval()

        # Sample random conditioning values from training data
        n_test = min(100, len(c_data))
        idx = np.random.choice(len(c_data), n_test, replace=False)
        c_test = c_data[idx]
        x_true = x_data[idx]

        # Generate predictions (mean of samples)
        x_pred = self.predict(c_test, n_samples=n_samples)

        # Compute metrics
        mae = np.mean(np.abs(x_pred - x_true))

        # R² score
        ss_res = np.sum((x_true - x_pred) ** 2)
        ss_tot = np.sum((x_true - np.mean(x_true, axis=0)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return mae, r2

    def plot_learning_curves(self, figsize=(12, 4)):
        """Plot training loss and accuracy curves."""
        import matplotlib.pyplot as plt

        if not self.history:
            print("No training history available. Run fit() first.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Loss curve
        axes[0].plot(self.history['loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        # MAE curve
        axes[1].plot(self.history['eval_epochs'], self.history['eval_mae'], 'o-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Reconstruction MAE')
        axes[1].grid(True, alpha=0.3)

        # R² curve
        axes[2].plot(self.history['eval_epochs'], self.history['eval_r2'], 'o-', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R² Score')
        axes[2].set_title('Reconstruction R²')
        axes[2].axhline(1.0, color='k', ls='--', alpha=0.3)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print final metrics
        print(f"Final Loss: {self.history['loss'][-1]:.6f}")
        print(f"Final MAE:  {self.history['eval_mae'][-1]:.6f}")
        print(f"Final R²:   {self.history['eval_r2'][-1]:.6f}")

    @torch.no_grad()
    def sample(self, c_values, n_samples=100, n_steps=50):
        """Generate samples conditioned on c_values.

        Parameters
        ----------
        c_values : ndarray, shape (c_dim,) or (n_conditions, c_dim)
            The conditioning values
        n_samples : int
            Number of samples per conditioning value
        n_steps : int
            Number of ODE integration steps

        Returns
        -------
        samples : ndarray
        """
        self.model.eval()

        c_values = np.atleast_2d(c_values)
        n_conditions = len(c_values)

        all_samples = []

        for c_val in c_values:
            # Normalize conditioning
            c = torch.tensor(c_val, dtype=torch.float32, device=self.device)
            c = (c - self.c_mean) / (self.c_std + 1e-8)
            c = c.unsqueeze(0).expand(n_samples, -1)

            # Start from noise
            x = torch.randn(n_samples, self.x_dim, device=self.device)

            # Integrate ODE from t=0 to t=1
            dt = 1.0 / n_steps
            for step in range(n_steps):
                t = torch.tensor(step / n_steps, device=self.device)
                v = self.model(x, t, c)
                x = x + v * dt

            # Unnormalize
            x = self._unnormalize_x(x)
            all_samples.append(x.cpu().numpy())

        if n_conditions == 1:
            return all_samples[0]
        return all_samples

    @torch.no_grad()
    def predict(self, c_values, n_samples=100):
        """Generate samples and return mean prediction."""
        samples = self.sample(c_values, n_samples=n_samples)
        if isinstance(samples, list):
            return np.array([s.mean(axis=0) for s in samples])
        return samples.mean(axis=0)


# =============================================================================
# Unconditional Flow Matching (for visualization)
# =============================================================================

class UnconditionalVelocityNetwork(nn.Module):
    """Neural network that predicts velocity given position and time (no conditioning)."""

    def __init__(self, dim=2, hidden_dim=128, n_layers=3):
        super().__init__()

        layers = [nn.Linear(dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """Predict velocity at (x, t)."""
        t = t.view(-1, 1) if t.dim() == 1 else t
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def train_unconditional_flow_matching(sample_source, sample_target,
                                       n_samples=5000, epochs=1000,
                                       hidden_dim=128, lr=1e-3,
                                       verbose=True, device_override=None):
    """Train an unconditional flow matching model.

    Parameters
    ----------
    sample_source : callable
        Function that returns n_samples from source distribution
    sample_target : callable
        Function that returns n_samples from target distribution
    n_samples : int
        Number of training samples
    epochs : int
        Training epochs
    hidden_dim : int
        Hidden layer dimension
    lr : float
        Learning rate
    verbose : bool
        Show progress
    device_override : torch.device, optional
        Device to use

    Returns
    -------
    model : UnconditionalVelocityNetwork
        Trained model
    losses : list
        Training losses
    """
    dev = device_override if device_override is not None else device

    model = UnconditionalVelocityNetwork(dim=2, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Generate paired samples
    x0 = torch.tensor(sample_source(n_samples), dtype=torch.float32, device=dev)
    x1 = torch.tensor(sample_target(n_samples), dtype=torch.float32, device=dev)

    losses = []
    pbar = tqdm(range(epochs), desc='Training', disable=not verbose)

    for epoch in pbar:
        optimizer.zero_grad()

        # Sample random times
        t = torch.rand(n_samples, 1, device=dev)

        # Interpolate: x_t = (1-t)*x0 + t*x1
        x_t = (1 - t) * x0 + t * x1

        # Target velocity: dx/dt = x1 - x0 (constant for linear interpolation)
        v_target = x1 - x0

        # Predicted velocity
        v_pred = model(x_t, t)

        # MSE loss
        loss = ((v_pred - v_target) ** 2).mean()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return model, losses
