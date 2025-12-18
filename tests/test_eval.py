"""Tests for MDN evaluation script."""
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

from src.ml_init.dataset import generate_dataset
from src.ml_init.train import train_mdn_model
from src.ml_init.eval import evaluate_mdn_model, load_model_and_metadata


def test_load_model_and_metadata():
    """Test loading model and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Generate small dataset and train
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
        
        # Load model and metadata
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        model, metadata, z = load_model_and_metadata(model_path)
        
        assert model is not None
        assert metadata is not None
        assert z is not None
        
        # Check metadata
        assert metadata["N_model"] == 64
        assert metadata["K_model"] == 5
        assert metadata["z_min"] == -8.0
        assert metadata["z_max"] == 8.0


def test_evaluate_mdn_model_basic():
    """Test basic model evaluation."""
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
        
        # Evaluate
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        test_data_path = data_dir / "test.npz"
        results = evaluate_mdn_model(model_path, test_data_path)
        
        # Check results structure
        assert "mean_ce" in results
        assert "mean_pdf_linf" in results
        assert "mean_cdf_linf" in results
        assert "quantile_errors" in results
        
        # Check values are reasonable
        assert results["mean_ce"] >= 0
        assert results["mean_pdf_linf"] >= 0
        assert results["mean_cdf_linf"] >= 0
        assert len(results["quantile_errors"]) > 0


def test_evaluate_mdn_model_save_results():
    """Test saving evaluation results to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        results_path = Path(tmpdir) / "results.json"
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
        
        # Evaluate and save
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        test_data_path = data_dir / "test.npz"
        results = evaluate_mdn_model(model_path, test_data_path, output_path=results_path)
        
        # Check file exists
        assert results_path.exists()
        
        # Check file content
        with open(results_path) as f:
            saved_results = json.load(f)
        
        assert saved_results["mean_ce"] == results["mean_ce"]
        assert saved_results["mean_pdf_linf"] == results["mean_pdf_linf"]


def test_evaluate_mdn_model_version_mismatch():
    """Test error handling for version mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Generate dataset and train
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
        
        # Corrupt metadata (change K)
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["K_model"] = 10  # Wrong K
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Should raise error
        model_path = output_dir / "mdn_init_v1_N64_K5.pt"
        test_data_path = data_dir / "test.npz"
        
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_mdn_model(model_path, test_data_path)

