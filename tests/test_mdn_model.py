"""Tests for MDN model."""
import numpy as np
import pytest
import torch
import torch.nn as nn

# Skip tests if PyTorch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from src.ml_init.model import MDNModel, log_gmm_pdf


def test_mdn_model_initialization():
    """Test MDN model initialization."""
    N = 64
    K = 5
    H = 128
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    
    assert model.N == N
    assert model.K == K
    assert model.H == H
    assert model.sigma_min == 1e-3


def test_mdn_model_forward():
    """Test MDN model forward pass."""
    N = 64
    K = 5
    H = 128
    batch_size = 4
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    model.eval()
    
    # Create input
    x = torch.randn(batch_size, N)
    
    # Forward pass
    alpha, mu, beta = model(x)
    
    # Check shapes
    assert alpha.shape == (batch_size, K)
    assert mu.shape == (batch_size, K)
    assert beta.shape == (batch_size, K)


def test_mdn_model_output_to_gmm_params():
    """Test conversion from model output to GMM parameters."""
    N = 64
    K = 5
    H = 128
    batch_size = 2
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    model.eval()
    
    x = torch.randn(batch_size, N)
    alpha, mu, beta = model(x)
    
    # Convert to GMM parameters
    pi = torch.softmax(alpha, dim=-1)
    sigma = torch.nn.functional.softplus(beta) + model.sigma_min
    
    # Check constraints
    assert torch.allclose(pi.sum(dim=-1), torch.ones(batch_size))
    assert torch.all(pi >= 0)
    assert torch.all(sigma > 0)
    assert torch.all(sigma >= model.sigma_min)


def test_mdn_model_sorting():
    """Test that outputs can be sorted by mu."""
    N = 64
    K = 5
    H = 128
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    model.eval()
    
    x = torch.randn(1, N)
    alpha, mu, beta = model(x)
    
    # Convert to GMM parameters
    pi = torch.softmax(alpha, dim=-1)
    sigma = torch.nn.functional.softplus(beta) + model.sigma_min
    
    # Sort by mu (ascending) - lexsort: (sigma, -pi, mu)
    mu_np = mu[0].detach().numpy()
    pi_np = pi[0].detach().numpy()
    sigma_np = sigma[0].detach().numpy()
    
    # Sort indices using lexsort (last key is primary)
    idx = np.lexsort((sigma_np, -pi_np, mu_np))
    mu_sorted = mu_np[idx]
    
    # Check that sorted means are ascending
    assert np.all(np.diff(mu_sorted) >= 0), "Sorted means should be ascending"


def test_log_gmm_pdf_basic():
    """Test log GMM PDF computation."""
    N = 64
    K = 3
    
    z = torch.linspace(-2, 2, N)
    pi = torch.tensor([0.3, 0.4, 0.3])
    mu = torch.tensor([-1.0, 0.0, 1.0])
    sigma = torch.tensor([0.5, 0.5, 0.5])
    
    log_f = log_gmm_pdf(z, pi, mu, sigma)
    
    assert log_f.shape == (N,)
    assert torch.all(torch.isfinite(log_f))
    
    # Convert to PDF and check normalization (approximately)
    f = torch.exp(log_f)
    w = torch.ones(N) * (z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    integral = (f * w).sum()
    assert torch.isclose(integral, torch.tensor(1.0), rtol=0.1)


def test_log_gmm_pdf_batch():
    """Test log GMM PDF computation with batch."""
    N = 64
    K = 3
    batch_size = 2
    
    z = torch.linspace(-2, 2, N)
    pi = torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
    mu = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])
    sigma = torch.tensor([[0.5, 0.5, 0.5], [0.4, 0.6, 0.4]])
    
    log_f = log_gmm_pdf(z, pi, mu, sigma)
    
    assert log_f.shape == (batch_size, N)
    assert torch.all(torch.isfinite(log_f))


def test_log_gmm_pdf_numerical_stability():
    """Test log GMM PDF with extreme values."""
    N = 64
    K = 3
    
    z = torch.linspace(-10, 10, N)
    pi = torch.tensor([0.33, 0.34, 0.33])
    mu = torch.tensor([-5.0, 0.0, 5.0])
    sigma = torch.tensor([0.1, 0.1, 0.1])  # Very narrow
    
    log_f = log_gmm_pdf(z, pi, mu, sigma)
    
    # Should not have NaN or Inf
    assert torch.all(torch.isfinite(log_f))


def test_mdn_model_gradient_flow():
    """Test that gradients flow through the model."""
    N = 64
    K = 5
    H = 128
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    model.train()
    
    x = torch.randn(2, N, requires_grad=True)
    alpha, mu, beta = model(x)
    
    # Compute loss (use a more meaningful loss that depends on x)
    # Loss: sum of means (ensures gradient flows)
    loss = mu.sum()
    
    # Backward
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_mdn_model_device():
    """Test model on CPU (device handling)."""
    N = 64
    K = 5
    H = 128
    
    model = MDNModel(N=N, K=K, H=H, sigma_min=1e-3)
    model.eval()
    
    x = torch.randn(2, N)
    alpha, mu, beta = model(x)
    
    # Should work on CPU
    assert alpha.device.type == 'cpu'
    assert mu.device.type == 'cpu'
    assert beta.device.type == 'cpu'

