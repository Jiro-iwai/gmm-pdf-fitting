"""Tests for MDN training script."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
import json

# Skip tests if PyTorch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from src.ml_init.dataset import generate_dataset
from src.ml_init.train import train_mdn_model, load_dataset


def test_load_dataset():
    """Test dataset loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Generate small dataset
        generate_dataset(
            output_dir=data_dir,
            n_train=100,
            n_val=20,
            n_test=20,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Load dataset
        train_loader, val_loader, test_loader, z = load_dataset(
            data_dir=data_dir,
            batch_size=32,
            num_workers=0,
        )
        
        # Check shapes
        assert z.shape == (64,)
        
        # Check loaders
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2  # (z, f)
        assert train_batch[1].shape[0] <= 32  # batch_size
        
        val_batch = next(iter(val_loader))
        assert len(val_batch) == 2


def test_train_mdn_model_basic():
    """Test basic training functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Generate small dataset
        generate_dataset(
            output_dir=data_dir,
            n_train=100,
            n_val=20,
            n_test=20,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Train for 1 epoch
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=32,
            lr=1e-3,
            epochs=1,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Check checkpoint exists
        checkpoint_files = list(output_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0, "Checkpoint should be created"
        
        # Check metadata exists
        metadata_file = output_dir / "metadata.json"
        assert metadata_file.exists(), "Metadata should be created"
        
        # Check metadata content
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["version"] == "mdn_init_v1"
        assert metadata["N_model"] == 64
        assert metadata["K_model"] == 5
        assert metadata["z_min"] == -8.0
        assert metadata["z_max"] == 8.0
        assert metadata["sigma_min"] == 1e-3
        assert "best_epoch" in metadata
        assert "best_val_ce" in metadata


def test_train_mdn_model_multiple_epochs():
    """Test training for multiple epochs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
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
        
        # Train for 3 epochs
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=16,
            lr=1e-3,
            epochs=3,
            lambda_mom=0.0,
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Check metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["best_epoch"] >= 0
        assert metadata["best_epoch"] < 3
        assert metadata["best_val_ce"] >= 0


def test_train_mdn_model_gradient_clipping():
    """Test that gradient clipping is applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        generate_dataset(
            output_dir=data_dir,
            n_train=20,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Training should complete without error (gradient clipping prevents explosion)
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
            max_grad_norm=1.0,
        )
        
        # Should complete successfully
        assert (output_dir / "metadata.json").exists()


def test_train_mdn_model_moment_penalty():
    """Test training with moment penalty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "checkpoints"
        data_dir.mkdir()
        output_dir.mkdir()
        
        generate_dataset(
            output_dir=data_dir,
            n_train=20,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Train with moment penalty
        train_mdn_model(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=8,
            lr=1e-3,
            epochs=1,
            lambda_mom=1.0,  # Non-zero penalty
            N=64,
            K=5,
            H=128,
            sigma_min=1e-3,
            num_workers=0,
        )
        
        # Should complete successfully
        assert (output_dir / "metadata.json").exists()

