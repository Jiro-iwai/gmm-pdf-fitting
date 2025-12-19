"""MDN (Mixture Density Network) model for GMM initialization."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MDNModel(nn.Module):
    """
    Mixture Density Network for GMM initialization.
    
    MLP that outputs GMM parameters (mixing weights, means, variances).
    """
    
    def __init__(
        self,
        N: int = 64,
        K: int = 5,
        H: int = 128,
        sigma_min: float = 1e-3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize MDN model.
        
        Parameters:
        -----------
        N : int
            Input dimension (number of grid points)
        K : int
            Number of GMM components
        H : int
            Hidden layer dimension
        sigma_min : float
            Minimum standard deviation (sqrt(reg_var))
        num_layers : int
            Number of hidden layers (2 or 3)
        dropout : float
            Dropout probability (0.0 = no dropout)
        """
        super().__init__()
        
        self.N = N
        self.K = K
        self.H = H
        self.sigma_min = sigma_min
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # Build layers dynamically
        layers = []
        in_dim = N
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, H))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = H
        
        self.hidden = nn.Sequential(*layers)
        
        # Output layer: H -> 3K (alpha, mu, beta)
        self.fc_out = nn.Linear(H, 3 * K)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He (ReLU layers) and Xavier (output layer)."""
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Xavier initialization for output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input PDF values, shape (batch_size, N)
        
        Returns:
        --------
        alpha : torch.Tensor
            Mixing weight logits, shape (batch_size, K)
        mu : torch.Tensor
            Component means, shape (batch_size, K)
        beta : torch.Tensor
            Variance parameter logits, shape (batch_size, K)
        """
        # Hidden layers
        h = self.hidden(x)
        
        # Output: 3K values
        output = self.fc_out(h)
        
        # Split into alpha, mu, beta
        alpha = output[:, :self.K]
        mu = output[:, self.K:2*self.K]
        beta = output[:, 2*self.K:]
        
        return alpha, mu, beta


def log_gmm_pdf(
    z: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log GMM PDF using log-sum-exp for numerical stability.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,) or (batch_size, N)
    pi : torch.Tensor
        Mixing weights, shape (K,) or (batch_size, K)
    mu : torch.Tensor
        Component means, shape (K,) or (batch_size, K)
    sigma : torch.Tensor
        Component standard deviations, shape (K,) or (batch_size, K)
    
    Returns:
    --------
    log_f : torch.Tensor
        Log PDF values, shape (N,) or (batch_size, N)
    """
    # Handle different input shapes
    if z.dim() == 1:
        # Single z, check if pi/mu/sigma are also 1D
        if pi.dim() == 1:
            z = z.unsqueeze(0)  # (1, N)
            pi = pi.unsqueeze(0)  # (1, K)
            mu = mu.unsqueeze(0)  # (1, K)
            sigma = sigma.unsqueeze(0)  # (1, K)
            squeeze_output = True
        else:
            # z is 1D but pi/mu/sigma are 2D (batch)
            # Broadcast z to match batch size
            batch_size = pi.shape[0]
            z = z.unsqueeze(0).expand(batch_size, -1)  # (batch_size, N)
            squeeze_output = False
    else:
        squeeze_output = False
    
    batch_size, N = z.shape
    K = pi.shape[1]
    
    # Expand dimensions for broadcasting
    # z: (batch_size, N, 1)
    # mu: (batch_size, 1, K)
    # sigma: (batch_size, 1, K)
    z_expanded = z.unsqueeze(-1)  # (batch_size, N, 1)
    mu_expanded = mu.unsqueeze(1)  # (batch_size, 1, K)
    sigma_expanded = sigma.unsqueeze(1)  # (batch_size, 1, K)
    
    # Compute log N(z; mu_k, sigma_k^2) for each component
    # log N = -0.5 * log(2π) - log(sigma) - 0.5 * ((z - mu) / sigma)^2
    log_2pi = np.log(2 * np.pi)
    log_normal = (
        -0.5 * log_2pi
        - torch.log(sigma_expanded)
        - 0.5 * ((z_expanded - mu_expanded) / sigma_expanded) ** 2
    )  # (batch_size, N, K)
    
    # Add log mixing weights
    log_pi = torch.log(pi + 1e-12)  # (batch_size, K)
    log_pi_expanded = log_pi.unsqueeze(1)  # (batch_size, 1, K)
    
    log_components = log_normal + log_pi_expanded  # (batch_size, N, K)
    
    # Log-sum-exp: log(sum_k exp(a_k)) = m + log(sum_k exp(a_k - m))
    # where m = max_k a_k
    m = torch.max(log_components, dim=-1, keepdim=True)[0]  # (batch_size, N, 1)
    log_f = m.squeeze(-1) + torch.log(
        torch.sum(torch.exp(log_components - m), dim=-1) + 1e-12
    )  # (batch_size, N)
    
    if squeeze_output:
        log_f = log_f.squeeze(0)  # (N,)
    
    return log_f

