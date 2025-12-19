"""Tests for MDN inference API."""
import numpy as np
import pytest
import tempfile
import json
from pathlib import Path

# Skip tests if PyTorch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from src.ml_init.infer import (
    MDNInitError,
    mdn_predict_init,
    _sort_components,
    _resample_to_fixed_grid,
    clear_model_cache,
    get_cache_size,
)
from src.ml_init.dataset import generate_dataset
from src.ml_init.train import train_mdn_model


def test_mdn_init_error():
    """Test MDNInitError exception."""
    error = MDNInitError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, RuntimeError)


def test_mdn_predict_init_basic():
    """Test basic MDN prediction."""
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
        
        # Load test data
        test_data = np.load(data_dir / "test.npz")
        z = test_data["z"]
        f = test_data["f"][0]  # First sample
        
        # Predict
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        result = mdn_predict_init(z, f, K=5, model_path=model_path)
        
        # Check output structure
        assert "pi" in result
        assert "mu" in result
        assert "var" in result
        
        # Check shapes
        assert result["pi"].shape == (5,)
        assert result["mu"].shape == (5,)
        assert result["var"].shape == (5,)
        
        # Check constraints
        assert np.allclose(np.sum(result["pi"]), 1.0)
        assert np.all(result["pi"] >= 0)
        assert np.all(result["var"] > 0)
        assert np.all(result["var"] >= 1e-6)  # reg_var
        
        # Check mu is sorted
        assert np.all(np.diff(result["mu"]) >= 0)


def test_mdn_predict_init_different_grid():
    """Test prediction with different grid size."""
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
        
        # Use different grid (128 points instead of 64)
        z_different = np.linspace(-8, 8, 128)
        f_different = np.exp(-z_different**2)
        f_different = f_different / np.sum(f_different * (z_different[1] - z_different[0]))
        
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        result = mdn_predict_init(z_different, f_different, K=5, model_path=model_path)
        
        # Should still work (resampling)
        assert result["pi"].shape == (5,)
        assert np.allclose(np.sum(result["pi"]), 1.0)


def test_mdn_predict_init_file_not_found():
    """Test error handling for missing model file."""
    z = np.linspace(-8, 8, 64)
    f = np.exp(-z**2)
    f = f / np.sum(f * (z[1] - z[0]))
    
    model_path = Path("/nonexistent/path/model.pt")
    
    with pytest.raises(MDNInitError) as exc_info:
        mdn_predict_init(z, f, K=5, model_path=model_path)
    
    assert "not found" in str(exc_info.value).lower() or "FileNotFound" in str(exc_info.value)


def test_mdn_predict_init_k_mismatch():
    """Test error handling for K mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Train model with K=5
        generate_dataset(
            output_dir=data_dir,
            n_train=20,
            n_val=5,
            n_test=5,
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
            batch_size=8,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Try to use with K=3 (mismatch)
        test_data = np.load(data_dir / "test.npz")
        z = test_data["z"]
        f = test_data["f"][0]
        
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        
        with pytest.raises(MDNInitError) as exc_info:
            mdn_predict_init(z, f, K=3, model_path=model_path)
        
        assert "K" in str(exc_info.value) or "mismatch" in str(exc_info.value).lower()


def test_mdn_predict_init_nan_output():
    """Test error handling for NaN output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Train model
        generate_dataset(
            output_dir=data_dir,
            n_train=20,
            n_val=5,
            n_test=5,
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
            batch_size=8,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Use invalid input (all zeros)
        z = np.linspace(-8, 8, 64)
        f = np.zeros(64)
        
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        
        # Should either work (with fallback) or raise error
        try:
            result = mdn_predict_init(z, f, K=5, model_path=model_path)
            # If it works, check output is valid
            assert np.all(np.isfinite(result["pi"]))
            assert np.all(np.isfinite(result["mu"]))
            assert np.all(np.isfinite(result["var"]))
        except MDNInitError:
            # Error is also acceptable
            pass


def test_sort_components():
    """Test component sorting."""
    from src.ml_init.infer import _sort_components
    
    pi = np.array([0.2, 0.3, 0.1, 0.4])
    mu = np.array([1.0, -1.0, 0.5, -0.5])
    sigma = np.array([0.3, 0.5, 0.2, 0.4])
    
    pi_sorted, mu_sorted, sigma_sorted = _sort_components(pi, mu, sigma)
    
    # Check mu is sorted
    assert np.all(np.diff(mu_sorted) >= 0)
    
    # Check shapes preserved
    assert pi_sorted.shape == pi.shape
    assert mu_sorted.shape == mu.shape
    assert sigma_sorted.shape == sigma.shape


def test_resample_to_fixed_grid():
    """Test resampling to fixed grid."""
    from src.ml_init.infer import _resample_to_fixed_grid
    
    # Original grid (different size)
    z_orig = np.linspace(-10, 10, 100)
    f_orig = np.exp(-z_orig**2)
    f_orig = f_orig / np.sum(f_orig * (z_orig[1] - z_orig[0]))
    
    # Target grid
    z_target = np.linspace(-8, 8, 64)
    
    f_resampled = _resample_to_fixed_grid(z_orig, f_orig, z_target)
    
    assert f_resampled.shape == z_target.shape
    assert np.all(f_resampled >= 0)
    
    # Should be approximately normalized
    w = np.full(64, z_target[1] - z_target[0])
    w[0] = w[-1] = (z_target[1] - z_target[0]) / 2
    integral = np.sum(f_resampled * w)
    assert np.isclose(integral, 1.0, rtol=0.1)


def test_model_cache():
    """Test that models are cached and reused."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Clear cache before test
        clear_model_cache()
        assert get_cache_size() == 0
        
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
        
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        
        # Setup PDF
        z = np.linspace(-8, 8, 64)
        f_true = np.exp(-0.5 * z**2)
        f_true = f_true / np.sum(f_true * (z[1] - z[0]))
        
        # First call: should load and cache
        result1 = mdn_predict_init(
            z, f_true, K=5,
            model_path=model_path,
            device="cpu",
        )
        assert get_cache_size() == 1
        
        # Second call: should use cache
        result2 = mdn_predict_init(
            z, f_true, K=5,
            model_path=model_path,
            device="cpu",
        )
        assert get_cache_size() == 1  # Still 1, not 2
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result1["pi"], result2["pi"])
        np.testing.assert_array_almost_equal(result1["mu"], result2["mu"])
        np.testing.assert_array_almost_equal(result1["var"], result2["var"])
        
        # Clear cache
        clear_model_cache()
        assert get_cache_size() == 0
        
        # After clearing, should load again
        result3 = mdn_predict_init(
            z, f_true, K=5,
            model_path=model_path,
            device="cpu",
        )
        assert get_cache_size() == 1

