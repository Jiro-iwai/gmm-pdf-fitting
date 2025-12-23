"""LAMF (Learned Accelerated Mixture Fitter) model architecture.

This module implements the neural network components for LAMF:
- InitNet: Predicts initial GMM parameters from input PDF
- RefineBlock: Learnable refinement block combining EM statistics and gradients
- LAMFFitter: Main model combining InitNet and T iterations of RefineBlock
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ==============================================================================
# Parameter Constraint Transformations (Section 4 of design doc)
# ==============================================================================

class ParameterTransform:
    """
    Transform between unconstrained and constrained GMM parameters.
    
    Unconstrained parameters:
        - alpha: (batch_size, K) - logits for mixing weights
        - c: (batch_size, 1) - base mean
        - beta: (batch_size, K-1) - delta logits for ordered means
        - gamma: (batch_size, K) - logits for standard deviations
    
    Constrained parameters:
        - pi: (batch_size, K) - mixing weights, sum to 1, >= 0
        - mu: (batch_size, K) - ordered means, mu_1 < mu_2 < ... < mu_K
        - sigma: (batch_size, K) - standard deviations, > sigma_min
    """
    
    def __init__(self, sigma_min: float = 1e-3, delta_min: float = 1e-4, pi_min: float = 0.0):
        """
        Initialize parameter transform.
        
        Parameters:
        -----------
        sigma_min : float
            Minimum standard deviation
        delta_min : float
            Minimum gap between adjacent means
        pi_min : float
            Minimum mixing weight (0 = no constraint)
        """
        self.sigma_min = sigma_min
        self.delta_min = delta_min
        self.pi_min = pi_min
    
    def project(
        self,
        alpha: torch.Tensor,
        c: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform unconstrained parameters to constrained GMM parameters.
        
        Parameters:
        -----------
        alpha : torch.Tensor
            Mixing weight logits, shape (batch_size, K)
        c : torch.Tensor
            Base mean, shape (batch_size, 1)
        beta : torch.Tensor
            Delta logits for means, shape (batch_size, K-1)
        gamma : torch.Tensor
            Variance logits, shape (batch_size, K)
        
        Returns:
        --------
        pi : torch.Tensor
            Mixing weights, shape (batch_size, K)
        mu : torch.Tensor
            Ordered means, shape (batch_size, K)
        sigma : torch.Tensor
            Standard deviations, shape (batch_size, K)
        """
        # Mixing weights: softmax with optional minimum constraint
        pi = F.softmax(alpha, dim=-1)
        if self.pi_min > 0:
            pi = torch.clamp(pi, min=self.pi_min)
            pi = pi / pi.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Means: monotonic increasing parameterization
        # mu_1 = c, mu_k = mu_{k-1} + softplus(beta_k) + delta_min
        deltas = F.softplus(beta) + self.delta_min  # (batch_size, K-1)
        cumsum_deltas = torch.cumsum(deltas, dim=-1)  # (batch_size, K-1)
        
        # mu = [c, c + delta_1, c + delta_1 + delta_2, ...]
        batch_size = c.shape[0]
        K = alpha.shape[1]
        
        mu = torch.zeros(batch_size, K, device=c.device, dtype=c.dtype)
        mu[:, 0] = c.squeeze(-1)
        mu[:, 1:] = c + cumsum_deltas
        
        # Standard deviations: softplus + floor
        sigma = F.softplus(gamma) + self.sigma_min
        
        return pi, mu, sigma
    
    def unproject(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform constrained GMM parameters to unconstrained parameters.
        
        Note: This is approximate due to softmax and softplus being non-invertible
        in practice (log(0) issues). Use with care.
        
        Parameters:
        -----------
        pi : torch.Tensor
            Mixing weights, shape (batch_size, K)
        mu : torch.Tensor
            Means, shape (batch_size, K)
        sigma : torch.Tensor
            Standard deviations, shape (batch_size, K)
        
        Returns:
        --------
        alpha : torch.Tensor
            Mixing weight logits, shape (batch_size, K)
        c : torch.Tensor
            Base mean, shape (batch_size, 1)
        beta : torch.Tensor
            Delta logits, shape (batch_size, K-1)
        gamma : torch.Tensor
            Variance logits, shape (batch_size, K)
        """
        eps = 1e-12
        
        # Inverse softmax: log(pi) (up to constant)
        alpha = torch.log(pi + eps)
        
        # Means: extract c and deltas
        c = mu[:, 0:1]  # (batch_size, 1)
        deltas = mu[:, 1:] - mu[:, :-1]  # (batch_size, K-1)
        deltas = torch.clamp(deltas - self.delta_min, min=eps)
        
        # Inverse softplus: beta = log(exp(delta) - 1)
        # For numerical stability: if delta > 20, beta ≈ delta
        beta = torch.where(
            deltas > 20,
            deltas,
            torch.log(torch.exp(deltas) - 1 + eps)
        )
        
        # Inverse softplus for sigma
        sigma_shifted = torch.clamp(sigma - self.sigma_min, min=eps)
        gamma = torch.where(
            sigma_shifted > 20,
            sigma_shifted,
            torch.log(torch.exp(sigma_shifted) - 1 + eps)
        )
        
        return alpha, c, beta, gamma


# ==============================================================================
# Positional Encoding for V5 architecture
# ==============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for grid positions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int = 64, max_len: int = 256):
        super().__init__()
        
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor, shape (batch_size, seq_len, d_model)
        
        Returns:
        --------
        output : torch.Tensor
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using grid positions.
    """
    
    def __init__(self, N: int = 96, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(N, d_model)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional encoding.
        
        Returns:
        --------
        pe : torch.Tensor
            Positional encoding, shape (batch_size, N, d_model)
        """
        positions = torch.arange(self.embedding.num_embeddings, device=device)
        pe = self.embedding(positions)  # (N, d_model)
        return pe.unsqueeze(0).expand(batch_size, -1, -1)


# ==============================================================================
# InitNet: Initial parameter prediction (Section 5.1)
# ==============================================================================

class InitNet(nn.Module):
    """
    Initial parameter predictor network.
    
    Takes PDF mass values w and outputs unconstrained GMM parameters.
    """
    
    def __init__(
        self,
        N: int = 96,
        K: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_log_input: bool = True,
        use_cdf_input: bool = True,
    ):
        """
        Initialize InitNet.
        
        Parameters:
        -----------
        N : int
            Number of grid points
        K : int
            Number of GMM components
        hidden_dim : int
            Hidden layer dimension
        num_layers : int
            Number of hidden layers
        dropout : float
            Dropout probability
        use_log_input : bool
            If True, include log(w + eps) as input feature
        use_cdf_input : bool
            If True, include cumulative sum (CDF) as input feature
        """
        super().__init__()
        
        self.N = N
        self.K = K
        self.use_log_input = use_log_input
        self.use_cdf_input = use_cdf_input
        
        # Calculate input dimension
        input_dim = N  # base: w
        if use_log_input:
            input_dim += N  # log(w + eps)
        if use_cdf_input:
            input_dim += N  # cumsum(w)
        
        # Build MLP
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # Output heads for unconstrained parameters
        # alpha: K values (mixing logits)
        # c: 1 value (base mean)
        # beta: K-1 values (delta logits)
        # gamma: K values (variance logits)
        self.fc_alpha = nn.Linear(hidden_dim, K)
        self.fc_c = nn.Linear(hidden_dim, 1)
        self.fc_beta = nn.Linear(hidden_dim, K - 1)
        self.fc_gamma = nn.Linear(hidden_dim, K)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU layers."""
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Output layers: Xavier initialization
        for fc in [self.fc_alpha, self.fc_c, self.fc_beta, self.fc_gamma]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
    
    def forward(
        self,
        w: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        -----------
        w : torch.Tensor
            PDF mass values, shape (batch_size, N), sum to 1
        z : torch.Tensor, optional
            Grid points, shape (N,) - not used but kept for interface consistency
        
        Returns:
        --------
        alpha : torch.Tensor
            Mixing weight logits, shape (batch_size, K)
        c : torch.Tensor
            Base mean, shape (batch_size, 1)
        beta : torch.Tensor
            Delta logits, shape (batch_size, K-1)
        gamma : torch.Tensor
            Variance logits, shape (batch_size, K)
        """
        eps = 1e-12
        
        # Build input features
        features = [w]
        
        if self.use_log_input:
            log_w = torch.log(w + eps)
            features.append(log_w)
        
        if self.use_cdf_input:
            cdf_w = torch.cumsum(w, dim=-1)
            features.append(cdf_w)
        
        x = torch.cat(features, dim=-1)
        
        # Forward through MLP
        h = self.hidden(x)
        
        # Output unconstrained parameters
        alpha = self.fc_alpha(h)
        c = self.fc_c(h)
        beta = self.fc_beta(h)
        gamma = self.fc_gamma(h)
        
        return alpha, c, beta, gamma


# ==============================================================================
# InitNetV2: Enhanced InitNet with Attention (V5 architecture)
# ==============================================================================

class InitNetV2(nn.Module):
    """
    Enhanced InitNet with positional encoding and self-attention.
    
    Architecture:
    1. Per-point features: w, log(w), cdf(w) -> MLP -> point embeddings
    2. Add positional encoding
    3. Self-attention layers to capture global context
    4. Pool and predict GMM parameters
    """
    
    def __init__(
        self,
        N: int = 96,
        K: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_attention_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_log_input: bool = True,
        use_cdf_input: bool = True,
        pe_type: str = "sinusoidal",  # "sinusoidal", "learned", or "none"
    ):
        """
        Initialize InitNetV2.
        
        Parameters:
        -----------
        N : int
            Number of grid points
        K : int
            Number of GMM components
        hidden_dim : int
            Hidden layer dimension
        num_layers : int
            Number of MLP layers for point embedding
        num_attention_layers : int
            Number of self-attention layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        use_log_input : bool
            If True, include log(w + eps) as input feature
        use_cdf_input : bool
            If True, include cumulative sum (CDF) as input feature
        pe_type : str
            Type of positional encoding: "sinusoidal", "learned", or "none"
        """
        super().__init__()
        
        self.N = N
        self.K = K
        self.use_log_input = use_log_input
        self.use_cdf_input = use_cdf_input
        self.pe_type = pe_type
        
        # Input features per grid point: w, (log_w), (cdf_w)
        features_per_point = 1
        if use_log_input:
            features_per_point += 1
        if use_cdf_input:
            features_per_point += 1
        
        # Point embedding MLP (per-point features -> hidden_dim)
        self.point_embed = nn.Sequential(
            nn.Linear(features_per_point, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        if pe_type == "sinusoidal":
            self.pe = SinusoidalPositionalEncoding(d_model=hidden_dim, max_len=N)
        elif pe_type == "learned":
            self.pe = LearnedPositionalEncoding(N=N, d_model=hidden_dim)
        else:
            self.pe = None
        
        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)
        
        # Global pooling + MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        for _ in range(num_layers - 1):
            self.global_mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.global_mlp.append(nn.LayerNorm(hidden_dim))
            self.global_mlp.append(nn.ReLU())
            self.global_mlp.append(nn.Dropout(dropout))
        
        # Output heads for unconstrained parameters
        self.fc_alpha = nn.Linear(hidden_dim, K)
        self.fc_c = nn.Linear(hidden_dim, 1)
        self.fc_beta = nn.Linear(hidden_dim, K - 1)
        self.fc_gamma = nn.Linear(hidden_dim, K)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        # Point embedding
        for module in self.point_embed:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Global MLP
        for module in self.global_mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Output layers: Xavier initialization
        for fc in [self.fc_alpha, self.fc_c, self.fc_beta, self.fc_gamma]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
    
    def forward(
        self,
        w: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        -----------
        w : torch.Tensor
            PDF mass values, shape (batch_size, N), sum to 1
        z : torch.Tensor, optional
            Grid points, shape (N,) - can be used for position encoding
        
        Returns:
        --------
        alpha, c, beta, gamma : torch.Tensor
            Unconstrained GMM parameters
        """
        batch_size, N = w.shape
        eps = 1e-12
        device = w.device
        
        # Build per-point features: (batch_size, N, features_per_point)
        features = [w.unsqueeze(-1)]  # (batch_size, N, 1)
        
        if self.use_log_input:
            log_w = torch.log(w + eps).unsqueeze(-1)
            features.append(log_w)
        
        if self.use_cdf_input:
            cdf_w = torch.cumsum(w, dim=-1).unsqueeze(-1)
            features.append(cdf_w)
        
        x = torch.cat(features, dim=-1)  # (batch_size, N, features_per_point)
        
        # Point embedding
        x = self.point_embed(x)  # (batch_size, N, hidden_dim)
        
        # Add positional encoding
        if self.pe is not None:
            if self.pe_type == "sinusoidal":
                x = self.pe(x)
            else:  # learned
                x = x + self.pe(batch_size, device)
        
        # Self-attention
        x = self.attention(x)  # (batch_size, N, hidden_dim)
        
        # Global pooling: mean + max
        x_mean = x.mean(dim=1)  # (batch_size, hidden_dim)
        x_max = x.max(dim=1)[0]  # (batch_size, hidden_dim)
        h = torch.cat([x_mean, x_max], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # Global MLP
        h = self.global_mlp(h)  # (batch_size, hidden_dim)
        
        # Output unconstrained parameters
        alpha = self.fc_alpha(h)
        c = self.fc_c(h)
        beta = self.fc_beta(h)
        gamma = self.fc_gamma(h)
        
        return alpha, c, beta, gamma


# ==============================================================================
# RefineBlock: Learnable refinement (Section 6)
# ==============================================================================

class RefineBlock(nn.Module):
    """
    Learnable refinement block that combines EM-style statistics with learned updates.
    
    This block takes current GMM parameters and computes:
    1. E-step: Responsibilities with learnable temperature
    2. M-step candidates: EM-style sufficient statistics
    3. Learned correction: MLP predicts update based on EM candidates and gradients
    4. Damped update: Combines EM candidate with learned correction
    """
    
    def __init__(
        self,
        K: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        share_weights: bool = True,
        init_tau: float = 1.0,
        init_lambda: float = 0.5,
        sigma_min: float = 1e-3,
        sigma_max: float = 5.0,
        corr_scale: float = 0.5,
    ):
        """
        Initialize RefineBlock.
        
        Parameters:
        -----------
        K : int
            Number of GMM components
        hidden_dim : int
            Hidden dimension for update MLP
        num_layers : int
            Number of layers in update MLP
        share_weights : bool
            If True, use same MLP for all components
        init_tau : float
            Initial temperature for E-step softmax
        init_lambda : float
            Initial damping factor (0 = pure EM, 1 = pure learned)
        sigma_min : float
            Minimum standard deviation
        sigma_max : float
            Maximum standard deviation (prevents exp overflow)
        corr_scale : float
            Scale for bounded corrections (tanh output multiplier)
        """
        super().__init__()
        
        self.K = K
        self.share_weights = share_weights
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.corr_scale = corr_scale
        
        # Precompute log bounds for sigma
        self.log_sigma_min = np.log(sigma_min)
        self.log_sigma_max = np.log(sigma_max)
        
        # Learnable temperature (tau > 0)
        # tau = softplus(tau_param) + tau_min
        self.tau_param = nn.Parameter(torch.tensor(0.0))
        self.tau_min = 0.1
        
        # Learnable damping factor (lambda in [0, 1])
        # lambda = sigmoid(lambda_param)
        self.lambda_param = nn.Parameter(torch.tensor(0.0))
        
        # Initialize to target values
        with torch.no_grad():
            # tau_param such that softplus(tau_param) + tau_min = init_tau
            target_softplus = init_tau - self.tau_min
            self.tau_param.data = torch.log(torch.exp(torch.tensor(target_softplus)) - 1)
            # lambda_param such that sigmoid(lambda_param) = init_lambda
            self.lambda_param.data = torch.log(
                torch.tensor(init_lambda / (1 - init_lambda + 1e-8))
            )
        
        # Input features per component for update MLP:
        # - Current params: log(pi), mu, log(sigma) -> 3
        # - EM candidates: N_k, m_k, log(v_k) -> 3
        # - Delta to EM: delta_log_pi, delta_mu, delta_log_sigma -> 3
        # - Global features: mean, std of target -> 2
        # Total: 11 per component
        input_per_component = 11
        
        if share_weights:
            # Single MLP shared across all components
            layers = []
            in_dim = input_per_component
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.update_mlp = nn.Sequential(*layers)
            
            # Output: corrections for alpha, mu (via c/beta proxy), gamma
            # For simplicity, output delta for all unconstrained params per component
            # delta_alpha (1), delta_mu (1), delta_gamma (1) -> 3
            self.fc_out = nn.Linear(hidden_dim, 3)
        else:
            # Separate MLPs per component
            self.update_mlps = nn.ModuleList()
            self.fc_outs = nn.ModuleList()
            for _ in range(K):
                layers = []
                in_dim = input_per_component
                for _ in range(num_layers):
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    in_dim = hidden_dim
                self.update_mlps.append(nn.Sequential(*layers))
                self.fc_outs.append(nn.Linear(hidden_dim, 3))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        if self.share_weights:
            for module in self.update_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)
            # Small initialization for output to start close to EM
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)
        else:
            for mlp in self.update_mlps:
                for module in mlp:
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                        nn.init.zeros_(module.bias)
            for fc in self.fc_outs:
                nn.init.zeros_(fc.weight)
                nn.init.zeros_(fc.bias)
    
    @property
    def tau(self) -> torch.Tensor:
        """Get current temperature."""
        return F.softplus(self.tau_param) + self.tau_min
    
    @property
    def damping(self) -> torch.Tensor:
        """Get current damping factor (lambda)."""
        return torch.sigmoid(self.lambda_param)
    
    def compute_responsibilities(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute responsibilities (E-step) with learnable temperature.
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N)
        pi : torch.Tensor
            Mixing weights, shape (batch_size, K)
        mu : torch.Tensor
            Means, shape (batch_size, K)
        sigma : torch.Tensor
            Standard deviations, shape (batch_size, K)
        
        Returns:
        --------
        r : torch.Tensor
            Responsibilities, shape (batch_size, N, K)
        """
        batch_size = w.shape[0]
        N = z.shape[0]
        K = pi.shape[1]
        
        # Expand for broadcasting
        z_exp = z.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        mu_exp = mu.unsqueeze(1)  # (batch_size, 1, K)
        sigma_exp = sigma.unsqueeze(1)  # (batch_size, 1, K)
        
        # Log of each component's contribution
        # log N(z; mu_k, sigma_k^2) = -0.5*log(2π) - log(sigma) - 0.5*((z-mu)/sigma)^2
        log_2pi = np.log(2 * np.pi)
        log_normal = (
            -0.5 * log_2pi
            - torch.log(sigma_exp + 1e-12)
            - 0.5 * ((z_exp - mu_exp) / (sigma_exp + 1e-12)) ** 2
        )  # (batch_size, N, K)
        
        # Add log mixing weights
        log_pi = torch.log(pi + 1e-12).unsqueeze(1)  # (batch_size, 1, K)
        log_weighted = log_normal + log_pi  # (batch_size, N, K)
        
        # Apply temperature-scaled softmax for responsibilities
        # r_ik = softmax(log_weighted / tau)
        tau = self.tau
        r = F.softmax(log_weighted / tau, dim=-1)  # (batch_size, N, K)
        
        return r
    
    def compute_em_statistics(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute EM sufficient statistics (weighted by responsibilities).
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N)
        r : torch.Tensor
            Responsibilities, shape (batch_size, N, K)
        
        Returns:
        --------
        N_k : torch.Tensor
            Effective counts, shape (batch_size, K)
        m_k : torch.Tensor
            Weighted means, shape (batch_size, K)
        v_k : torch.Tensor
            Weighted variances, shape (batch_size, K)
        """
        eps = 1e-8
        
        # Weighted responsibilities
        w_exp = w.unsqueeze(-1)  # (batch_size, N, 1)
        weighted_r = w_exp * r  # (batch_size, N, K)
        
        # N_k = sum_i w_i * r_ik
        N_k = weighted_r.sum(dim=1)  # (batch_size, K)
        N_k = torch.clamp(N_k, min=eps)
        
        # m_k = (1/N_k) * sum_i w_i * r_ik * z_i
        z_exp = z.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        m_k = (weighted_r * z_exp).sum(dim=1) / N_k  # (batch_size, K)
        
        # v_k = (1/N_k) * sum_i w_i * r_ik * (z_i - m_k)^2
        m_k_exp = m_k.unsqueeze(1)  # (batch_size, 1, K)
        v_k = (weighted_r * (z_exp - m_k_exp) ** 2).sum(dim=1) / N_k  # (batch_size, K)
        v_k = torch.clamp(v_k, min=eps)
        
        return N_k, m_k, v_k
    
    def forward(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        global_mean: torch.Tensor,
        global_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute refined GMM parameters.
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N)
        pi : torch.Tensor
            Current mixing weights, shape (batch_size, K)
        mu : torch.Tensor
            Current means, shape (batch_size, K)
        sigma : torch.Tensor
            Current standard deviations, shape (batch_size, K)
        global_mean : torch.Tensor
            Target distribution mean, shape (batch_size, 1)
        global_std : torch.Tensor
            Target distribution std, shape (batch_size, 1)
        
        Returns:
        --------
        pi_new : torch.Tensor
            Refined mixing weights, shape (batch_size, K)
        mu_new : torch.Tensor
            Refined means, shape (batch_size, K)
        sigma_new : torch.Tensor
            Refined standard deviations, shape (batch_size, K)
        """
        batch_size = w.shape[0]
        K = pi.shape[1]
        eps = 1e-12
        
        # E-step: compute responsibilities
        r = self.compute_responsibilities(z, w, pi, mu, sigma)
        
        # M-step candidates (EM statistics)
        N_k, m_k, v_k = self.compute_em_statistics(z, w, r)
        
        # EM candidate parameters
        pi_em = N_k  # Will be normalized later
        mu_em = m_k
        sigma_em = torch.sqrt(v_k)
        
        # Build input features for update MLP
        # Per-component features
        log_pi = torch.log(pi + eps)
        log_sigma = torch.log(sigma + eps)
        log_v_k = torch.log(v_k + eps)
        
        # Delta to EM candidates
        delta_log_pi = torch.log(N_k + eps) - log_pi
        delta_mu = m_k - mu
        delta_log_sigma = 0.5 * log_v_k - log_sigma
        
        # Global features (broadcast to all components)
        global_mean_exp = global_mean.expand(batch_size, K)
        global_std_exp = global_std.expand(batch_size, K)
        
        # Stack features: (batch_size, K, 11)
        features = torch.stack([
            log_pi,           # Current log mixing weight
            mu,               # Current mean
            log_sigma,        # Current log std
            N_k,              # EM effective count
            m_k,              # EM mean
            log_v_k,          # EM log variance
            delta_log_pi,     # Delta log pi to EM
            delta_mu,         # Delta mu to EM
            delta_log_sigma,  # Delta log sigma to EM
            global_mean_exp,  # Global mean
            global_std_exp,   # Global std
        ], dim=-1)  # (batch_size, K, 11)
        
        # Compute learned corrections
        if self.share_weights:
            # Shared MLP: process all components together
            features_flat = features.view(batch_size * K, -1)  # (batch_size*K, 11)
            h = self.update_mlp(features_flat)
            corrections = self.fc_out(h)  # (batch_size*K, 3)
            corrections = corrections.view(batch_size, K, 3)  # (batch_size, K, 3)
        else:
            # Separate MLPs per component
            corrections_list = []
            for k in range(K):
                feat_k = features[:, k, :]  # (batch_size, 11)
                h_k = self.update_mlps[k](feat_k)
                corr_k = self.fc_outs[k](h_k)  # (batch_size, 3)
                corrections_list.append(corr_k)
            corrections = torch.stack(corrections_list, dim=1)  # (batch_size, K, 3)
        
        # Extract corrections for each parameter and bound with tanh
        # This prevents unbounded corrections that can destabilize training
        raw_corr_alpha = corrections[:, :, 0]  # (batch_size, K)
        raw_corr_mu = corrections[:, :, 1]     # (batch_size, K)
        raw_corr_gamma = corrections[:, :, 2]  # (batch_size, K)
        
        # Bound corrections using tanh * scale
        corr_alpha = self.corr_scale * torch.tanh(raw_corr_alpha)
        corr_mu = self.corr_scale * global_std * torch.tanh(raw_corr_mu)  # Scale by global_std
        corr_gamma = self.corr_scale * torch.tanh(raw_corr_gamma)
        
        # Damped update: blend EM candidate with learned correction
        # For pi: use log-space blending
        lam = self.damping
        
        # Log-space blend for pi
        log_pi_em = torch.log(N_k + eps)  # Unnormalized
        log_pi_blend = (1 - lam) * log_pi + lam * log_pi_em + corr_alpha
        pi_new = F.softmax(log_pi_blend, dim=-1)
        
        # Linear blend for mu
        mu_new = (1 - lam) * mu + lam * mu_em + corr_mu
        
        # Log-space blend for sigma with upper/lower bounds
        log_sigma_em = 0.5 * log_v_k
        log_sigma_blend = (1 - lam) * log_sigma + lam * log_sigma_em + corr_gamma
        # Clip log_sigma to prevent exp overflow/underflow
        log_sigma_blend = torch.clamp(log_sigma_blend, min=self.log_sigma_min, max=self.log_sigma_max)
        sigma_new = torch.exp(log_sigma_blend)
        
        return pi_new, mu_new, sigma_new


# ==============================================================================
# LAMFFitter: Main model (Section 5 + 6 combined)
# ==============================================================================

class LAMFFitter(nn.Module):
    """
    LAMF (Learned Accelerated Mixture Fitter) main model.
    
    Combines InitNet and T iterations of RefineBlock to fit GMM to input PDF.
    """
    
    def __init__(
        self,
        N: int = 96,
        K: int = 5,
        T: int = 6,
        init_hidden_dim: int = 256,
        init_num_layers: int = 3,
        refine_hidden_dim: int = 128,
        refine_num_layers: int = 2,
        sigma_min: float = 1e-3,
        sigma_max: float = 5.0,
        pi_min: float = 0.0,
        corr_scale: float = 0.5,
        dropout: float = 0.1,
        share_refine_weights: bool = True,
        # V5 architecture options
        use_attention: bool = False,
        num_attention_layers: int = 2,
        num_attention_heads: int = 4,
        pe_type: str = "sinusoidal",
    ):
        """
        Initialize LAMF model.
        
        Parameters:
        -----------
        N : int
            Number of grid points (must match training data)
        K : int
            Number of GMM components
        T : int
            Number of refinement iterations
        init_hidden_dim : int
            Hidden dimension for InitNet
        init_num_layers : int
            Number of layers in InitNet
        refine_hidden_dim : int
            Hidden dimension for RefineBlock
        refine_num_layers : int
            Number of layers in RefineBlock update MLP
        sigma_min : float
            Minimum standard deviation
        sigma_max : float
            Maximum standard deviation (prevents exp overflow)
        pi_min : float
            Minimum mixing weight (0 = no constraint)
        corr_scale : float
            Scale for bounded corrections in RefineBlock
        dropout : float
            Dropout probability
        share_refine_weights : bool
            If True, all RefineBlocks share weights
        use_attention : bool
            If True, use InitNetV2 with attention (V5 architecture)
        num_attention_layers : int
            Number of attention layers in InitNetV2
        num_attention_heads : int
            Number of attention heads in InitNetV2
        pe_type : str
            Positional encoding type for InitNetV2: "sinusoidal", "learned", "none"
        """
        super().__init__()
        
        self.N = N
        self.K = K
        self.T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.pi_min = pi_min
        self.corr_scale = corr_scale
        self.use_attention = use_attention
        
        # Parameter transform
        self.transform = ParameterTransform(sigma_min=sigma_min, pi_min=pi_min)
        
        # InitNet (V4 or V5 with attention)
        if use_attention:
            self.init_net = InitNetV2(
                N=N,
                K=K,
                hidden_dim=init_hidden_dim,
                num_layers=init_num_layers,
                num_attention_layers=num_attention_layers,
                num_heads=num_attention_heads,
                dropout=dropout,
                pe_type=pe_type,
            )
        else:
            self.init_net = InitNet(
                N=N,
                K=K,
                hidden_dim=init_hidden_dim,
                num_layers=init_num_layers,
                dropout=dropout,
            )
        
        # RefineBlocks
        if share_refine_weights:
            # Single RefineBlock, applied T times
            self.refine_block = RefineBlock(
                K=K,
                hidden_dim=refine_hidden_dim,
                num_layers=refine_num_layers,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                corr_scale=corr_scale,
            )
            self.refine_blocks = None
        else:
            # T separate RefineBlocks
            self.refine_block = None
            self.refine_blocks = nn.ModuleList([
                RefineBlock(
                    K=K,
                    hidden_dim=refine_hidden_dim,
                    num_layers=refine_num_layers,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    corr_scale=corr_scale,
                )
                for _ in range(T)
            ])
    
    def compute_global_stats(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global mean and std of target distribution.
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N)
        
        Returns:
        --------
        mean : torch.Tensor
            Mean, shape (batch_size, 1)
        std : torch.Tensor
            Standard deviation, shape (batch_size, 1)
        """
        # Mean: sum_i w_i * z_i
        mean = (w * z.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Variance: sum_i w_i * (z_i - mean)^2
        var = (w * (z.unsqueeze(0) - mean) ** 2).sum(dim=-1, keepdim=True)
        std = torch.sqrt(var + 1e-12)  # (batch_size, 1)
        
        return mean, std
    
    def forward(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
        return_intermediate: bool = False,
    ) -> dict:
        """
        Forward pass: fit GMM to input PDF.
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N), sum to 1
        return_intermediate : bool
            If True, return parameters at each iteration
        
        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'pi': Final mixing weights, shape (batch_size, K)
            - 'mu': Final means, shape (batch_size, K)
            - 'sigma': Final standard deviations, shape (batch_size, K)
            - 'intermediate': List of (pi, mu, sigma) at each iteration
                              (only if return_intermediate=True)
        """
        batch_size = w.shape[0]
        
        # Compute global statistics
        global_mean, global_std = self.compute_global_stats(z, w)
        
        # InitNet: predict initial unconstrained parameters
        alpha, c, beta, gamma = self.init_net(w, z)
        
        # Project to constrained parameters
        pi, mu, sigma = self.transform.project(alpha, c, beta, gamma)
        
        intermediate = []
        if return_intermediate:
            intermediate.append((pi.clone(), mu.clone(), sigma.clone()))
        
        # Refinement iterations
        for t in range(self.T):
            if self.refine_blocks is not None:
                refine_block = self.refine_blocks[t]
            else:
                refine_block = self.refine_block
            
            pi, mu, sigma = refine_block(
                z, w, pi, mu, sigma, global_mean, global_std
            )
            
            if return_intermediate:
                intermediate.append((pi.clone(), mu.clone(), sigma.clone()))
        
        result = {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
        }
        
        if return_intermediate:
            result['intermediate'] = intermediate
        
        return result
    
    def get_gmm_params(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get final GMM parameters (convenience method).
        
        Parameters:
        -----------
        z : torch.Tensor
            Grid points, shape (N,)
        w : torch.Tensor
            PDF mass values, shape (batch_size, N)
        
        Returns:
        --------
        pi : torch.Tensor
            Mixing weights, shape (batch_size, K)
        mu : torch.Tensor
            Means, shape (batch_size, K)
        sigma : torch.Tensor
            Standard deviations, shape (batch_size, K)
        """
        result = self.forward(z, w)
        return result['pi'], result['mu'], result['sigma']

