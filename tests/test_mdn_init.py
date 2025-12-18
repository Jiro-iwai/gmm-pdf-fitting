"""Integration tests for MDN initialization."""
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

# Skip tests if PyTorch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from src.gmm_fitting.em_method import fit_gmm1d_to_pdf_weighted_em
from src.gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid
from src.ml_init.dataset import generate_dataset
from src.ml_init.train import train_mdn_model


def test_init_mdn_basic():
    """Test basic MDN initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Generate dataset and train model
        generate_dataset(
            output_dir=data_dir,
            n_train=50,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=16,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Setup PDF
        z = np.linspace(-8, 8, 64)
        f_true = max_pdf_bivariate_normal(z, 0.1, 0.4, 0.15, 0.9, 0.9)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Fit with MDN initialization
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f_true,
            K=5,
            max_iter=10,  # Small number since MDN should give good init
            tol=1e-4,
            reg_var=1e-6,
            n_init=1,  # MDN init, so only 1 init needed
            seed=1,
            init="mdn",
            mdn_model_path=str(model_path),
            mdn_device="cpu",
        )
        
        # Check output
        assert params is not None
        assert len(params.pi) == 5
        assert len(params.mu) == 5
        assert len(params.var) == 5
        
        # Check constraints
        assert np.allclose(np.sum(params.pi), 1.0)
        assert np.all(params.pi >= 0)
        assert np.all(params.var > 0)
        assert np.all(params.var >= 1e-6)


def test_init_mdn_fallback_on_error():
    """Test that fallback works when MDN fails."""
    z = np.linspace(-8, 8, 64)
    f_true = max_pdf_bivariate_normal(z, 0.1, 0.4, 0.15, 0.9, 0.9)
    f_true = normalize_pdf_on_grid(z, f_true)
    
    # Use non-existent model path
    model_path = Path("/nonexistent/path/model.pt")
    
    # Should fallback to wqmi (or other fallback)
    params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
        z, f_true,
        K=5,
        max_iter=10,
        tol=1e-4,
        reg_var=1e-6,
        n_init=1,
        seed=1,
        init="mdn",
        mdn_model_path=str(model_path),
        mdn_device="cpu",
    )
    
    # Should still work (fallback)
    assert params is not None
    assert len(params.pi) == 5


def test_init_mdn_with_env_var():
    """Test MDN initialization with environment variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Generate dataset and train
        generate_dataset(
            output_dir=data_dir,
            n_train=50,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=16,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Set environment variable
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        os.environ["MDN_MODEL_PATH"] = str(model_path)
        
        try:
            z = np.linspace(-8, 8, 64)
            f_true = max_pdf_bivariate_normal(z, 0.1, 0.4, 0.15, 0.9, 0.9)
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Fit with MDN initialization (no model_path specified)
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true,
                K=5,
                max_iter=10,
                tol=1e-4,
                reg_var=1e-6,
                n_init=1,
                seed=1,
                init="mdn",
                mdn_model_path=None,  # Should use env var
                mdn_device="cpu",
            )
            
            assert params is not None
            assert len(params.pi) == 5
        finally:
            # Clean up
            if "MDN_MODEL_PATH" in os.environ:
                del os.environ["MDN_MODEL_PATH"]


def test_init_mdn_default_path():
    """Test MDN initialization with default path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        default_path = Path(tmpdir) / "ml_init" / "checkpoints" / "mdn_init_v1_N64_K5.pt"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset and train
        generate_dataset(
            output_dir=data_dir,
            n_train=50,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=16,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Copy model to default location
        import shutil
        shutil.copy(
            output_dir / "mdn_init_v1_N64_K5.pt",
            default_path
        )
        shutil.copy(
            output_dir / "metadata.json",
            default_path.parent / "metadata.json"
        )
        
        # Change to tmpdir to test relative path resolution
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            z = np.linspace(-8, 8, 64)
            f_true = max_pdf_bivariate_normal(z, 0.1, 0.4, 0.15, 0.9, 0.9)
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Fit with MDN initialization (no model_path, no env var)
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true,
                K=5,
                max_iter=10,
                tol=1e-4,
                reg_var=1e-6,
                n_init=1,
                seed=1,
                init="mdn",
                mdn_model_path=None,
                mdn_device="cpu",
            )
            
            assert params is not None
            assert len(params.pi) == 5
        finally:
            os.chdir(original_cwd)

