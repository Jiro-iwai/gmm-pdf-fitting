"""Training script for LAMF model.

This module provides training utilities for LAMF (Learned Accelerated Mixture Fitter).
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .model import LAMFFitter
from .dataset import create_dataloaders
from .metrics import (
    compute_cross_entropy_loss,
    compute_moment_loss,
    compute_deep_supervision_loss,
    compute_gmm_pdf,
    compute_pdf_linf_error,
    compute_cdf_linf_error,
)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA.
        
        Parameters:
        -----------
        model : nn.Module
            Model to track
        decay : float
            EMA decay rate (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict for saving."""
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


def train_one_epoch(
    model: LAMFFitter,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_mom: float = 0.0,
    lambda_pdf: float = 0.0,
    eta_schedule: str = "linear",
    grad_clip: float = 1.0,
) -> dict:
    """
    Train for one epoch.
    
    Parameters:
    -----------
    model : LAMFFitter
        LAMF model
    train_loader : DataLoader
        Training dataloader
    optimizer : Optimizer
        Optimizer
    device : torch.device
        Device to train on
    lambda_mom : float
        Weight for moment loss
    lambda_pdf : float
        Weight for PDF L2 loss (0 = CE only, 1 = PDF L2 only)
    eta_schedule : str
        Deep supervision weight schedule
    grad_clip : float
        Gradient clipping value
    
    Returns:
    --------
    metrics : dict
        Training metrics for this epoch
    """
    model.train()
    
    total_loss = 0.0
    total_ce = 0.0
    total_pdf = 0.0
    total_mom = 0.0
    n_batches = 0
    
    for batch in train_loader:
        z = batch['z'][0].to(device)  # (N,) - same for all samples
        w = batch['w'].to(device)  # (batch_size, N)
        f_true = batch['f'].to(device) if lambda_pdf > 0 else None  # (batch_size, N)
        
        optimizer.zero_grad()
        
        # Forward pass with intermediate outputs for deep supervision
        result = model(z, w, return_intermediate=True)
        intermediate = result['intermediate']
        
        # Compute deep supervision loss
        loss = compute_deep_supervision_loss(
            z, w, intermediate,
            lambda_mom=lambda_mom,
            lambda_pdf=lambda_pdf,
            f_true=f_true,
            eta_schedule=eta_schedule,
        )
        
        # NaN/Inf detection: skip batch if loss is invalid
        if not torch.isfinite(loss):
            print(f"  Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            final_pi, final_mu, final_sigma = result['pi'], result['mu'], result['sigma']
            ce = compute_cross_entropy_loss(z, w, final_pi, final_mu, final_sigma).mean()
            
            total_loss += loss.item()
            total_ce += ce.item()
            
            if lambda_pdf > 0 and f_true is not None:
                from .metrics import compute_pdf_l2_loss
                pdf = compute_pdf_l2_loss(z, f_true, final_pi, final_mu, final_sigma).mean()
                total_pdf += pdf.item()
            
            if lambda_mom > 0:
                mom = compute_moment_loss(z, w, final_pi, final_mu, final_sigma).mean()
                total_mom += mom.item()
        
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'ce': total_ce / n_batches,
        'pdf': total_pdf / n_batches if lambda_pdf > 0 else 0.0,
        'mom': total_mom / n_batches if lambda_mom > 0 else 0.0,
    }


@torch.no_grad()
def validate(
    model: LAMFFitter,
    val_loader: DataLoader,
    device: torch.device,
    lambda_mom: float = 0.0,
) -> dict:
    """
    Validate model.
    
    Parameters:
    -----------
    model : LAMFFitter
        LAMF model
    val_loader : DataLoader
        Validation dataloader
    device : torch.device
        Device
    lambda_mom : float
        Weight for moment loss
    
    Returns:
    --------
    metrics : dict
        Validation metrics
    """
    model.eval()
    
    total_ce = 0.0
    total_mom = 0.0
    total_pdf_linf = 0.0
    total_cdf_linf = 0.0
    n_samples = 0
    
    for batch in val_loader:
        z = batch['z'][0].to(device)  # (N,) - same for all samples
        w = batch['w'].to(device)
        f_true = batch['f'].to(device)
        
        batch_size = w.shape[0]
        
        # Forward pass
        result = model(z, w)
        pi, mu, sigma = result['pi'], result['mu'], result['sigma']
        
        # Cross-entropy
        ce = compute_cross_entropy_loss(z, w, pi, mu, sigma)
        total_ce += ce.sum().item()
        
        # Moment loss
        if lambda_mom > 0:
            mom = compute_moment_loss(z, w, pi, mu, sigma)
            total_mom += mom.sum().item()
        
        # PDF/CDF errors
        f_hat = compute_gmm_pdf(z, pi, mu, sigma)
        pdf_linf = compute_pdf_linf_error(z, f_true, f_hat)
        cdf_linf = compute_cdf_linf_error(z, f_true, f_hat)
        
        total_pdf_linf += pdf_linf.sum().item()
        total_cdf_linf += cdf_linf.sum().item()
        
        n_samples += batch_size
    
    return {
        'ce': total_ce / n_samples,
        'mom': total_mom / n_samples if lambda_mom > 0 else 0.0,
        'pdf_linf': total_pdf_linf / n_samples,
        'cdf_linf': total_cdf_linf / n_samples,
    }


def train_lamf(
    data_dir: str,
    output_dir: str,
    # Model hyperparameters
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
    # Training hyperparameters
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    lambda_mom: float = 0.0,
    lambda_pdf: float = 0.0,
    # Lambda PDF curriculum
    lambda_pdf_curriculum: bool = False,
    lambda_pdf_start_epoch: int = 5,
    lambda_pdf_end_epoch: int = 15,
    lambda_pdf_max: float = 0.1,
    eta_schedule: str = "linear",
    grad_clip: float = 1.0,
    # Scheduler
    use_scheduler: bool = True,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 5,
    # Early stopping
    patience: int = 10,
    min_delta: float = 1e-5,
    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    # Other
    seed: int = 42,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> dict:
    """
    Train LAMF model.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing train.npz, val.npz, test.npz
    output_dir : str
        Output directory for checkpoints and logs
    K : int
        Number of GMM components
    T : int
        Number of refinement iterations
    ... (other parameters as documented)
    
    Returns:
    --------
    result : dict
        Training result containing best metrics and model path
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Training on device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    N = metadata['N']
    print(f"Data: N={N}, train={metadata['n_train']}, val={metadata['n_val']}, test={metadata['n_test']}")
    
    # Create model
    model = LAMFFitter(
        N=N,
        K=K,
        T=T,
        init_hidden_dim=init_hidden_dim,
        init_num_layers=init_num_layers,
        refine_hidden_dim=refine_hidden_dim,
        refine_num_layers=refine_num_layers,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        pi_min=pi_min,
        corr_scale=corr_scale,
        dropout=dropout,
        share_refine_weights=share_refine_weights,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # EMA
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        print(f"Using EMA with decay={ema_decay}")
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=lr * 0.01,
            )
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.5,
            )
    
    # Training history
    history = {
        'train_loss': [],
        'train_ce': [],
        'val_ce': [],
        'val_pdf_linf': [],
        'val_cdf_linf': [],
        'lr': [],
    }
    
    # Best model tracking (use pdf_linf as criterion)
    best_val_pdf_linf = float('inf')
    best_val_ce = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    if lambda_pdf_curriculum:
        print(f"Lambda PDF curriculum: epoch {lambda_pdf_start_epoch} -> {lambda_pdf_end_epoch}, max={lambda_pdf_max}")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Warmup
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Calculate lambda_pdf for this epoch (curriculum or fixed)
        if lambda_pdf_curriculum:
            if epoch < lambda_pdf_start_epoch:
                current_lambda_pdf = 0.0
            elif epoch >= lambda_pdf_end_epoch:
                current_lambda_pdf = lambda_pdf_max
            else:
                # Linear interpolation
                progress = (epoch - lambda_pdf_start_epoch) / (lambda_pdf_end_epoch - lambda_pdf_start_epoch)
                current_lambda_pdf = lambda_pdf_max * progress
        else:
            current_lambda_pdf = lambda_pdf
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_mom=lambda_mom,
            lambda_pdf=current_lambda_pdf,
            eta_schedule=eta_schedule,
            grad_clip=grad_clip,
        )
        
        # Update EMA after each epoch
        if ema is not None:
            ema.update()
        
        # Validate (use EMA params if available)
        if ema is not None:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, device, lambda_mom=lambda_mom)
        if ema is not None:
            ema.restore()
        
        # Update scheduler (after warmup)
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_ce'].append(train_metrics['ce'])
        history['val_ce'].append(val_metrics['ce'])
        history['val_pdf_linf'].append(val_metrics['pdf_linf'])
        history['val_cdf_linf'].append(val_metrics['cdf_linf'])
        history['lr'].append(current_lr)
        
        # Check for improvement (based on pdf_linf, not CE)
        if val_metrics['pdf_linf'] < best_val_pdf_linf - min_delta:
            best_val_pdf_linf = val_metrics['pdf_linf']
            best_val_ce = val_metrics['ce']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model (use EMA params if available)
            save_state_dict = model.state_dict()
            if ema is not None:
                ema.apply_shadow()
                save_state_dict = model.state_dict()
                ema.restore()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ce': val_metrics['ce'],
                'val_pdf_linf': val_metrics['pdf_linf'],
                'val_cdf_linf': val_metrics['cdf_linf'],
                'ema_decay': ema_decay if use_ema else None,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        # Print progress
        epoch_time = time.time() - epoch_start
        curriculum_str = f" | λ_pdf: {current_lambda_pdf:.3f}" if lambda_pdf_curriculum else ""
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Train CE: {train_metrics['ce']:.4f} | "
            f"Val CE: {val_metrics['ce']:.4f} | "
            f"Val PDF L∞: {val_metrics['pdf_linf']:.4f} | "
            f"Val CDF L∞: {val_metrics['cdf_linf']:.4f} | "
            f"LR: {current_lr:.2e}{curriculum_str} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, device, lambda_mom=lambda_mom)
    print(f"Test CE: {test_metrics['ce']:.4f}")
    print(f"Test PDF L∞: {test_metrics['pdf_linf']:.4f}")
    print(f"Test CDF L∞: {test_metrics['cdf_linf']:.4f}")
    
    # Save final model and metadata
    torch.save(model.state_dict(), output_dir / "lamf_model.pt")
    
    # Save metadata
    model_metadata = {
        'N': N,
        'K': K,
        'T': T,
        'init_hidden_dim': init_hidden_dim,
        'init_num_layers': init_num_layers,
        'refine_hidden_dim': refine_hidden_dim,
        'refine_num_layers': refine_num_layers,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'pi_min': pi_min,
        'corr_scale': corr_scale,
        'dropout': dropout,
        'share_refine_weights': share_refine_weights,
        'z_min': metadata['z_min'],
        'z_max': metadata['z_max'],
        'best_epoch': best_epoch,
        'best_val_ce': best_val_ce,
        'test_ce': test_metrics['ce'],
        'test_pdf_linf': test_metrics['pdf_linf'],
        'test_cdf_linf': test_metrics['cdf_linf'],
        'training_time_minutes': total_time / 60,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir / "training_curve.png")
    
    return {
        'best_epoch': best_epoch,
        'best_val_ce': best_val_ce,
        'test_metrics': test_metrics,
        'model_path': str(output_dir / "lamf_model.pt"),
    }


def plot_training_curves(history: dict, output_path: Path) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CE
    ax = axes[0, 1]
    ax.plot(epochs, history['train_ce'], label='Train CE')
    ax.plot(epochs, history['val_ce'], label='Val CE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy')
    ax.set_title('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PDF L∞
    ax = axes[1, 0]
    ax.plot(epochs, history['val_pdf_linf'], label='Val PDF L∞', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDF L∞ Error')
    ax.set_title('Validation PDF L∞ Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CDF L∞
    ax = axes[1, 1]
    ax.plot(epochs, history['val_cdf_linf'], label='Val CDF L∞', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CDF L∞ Error')
    ax.set_title('Validation CDF L∞ Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LAMF model")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.npz, val.npz, test.npz")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Model
    parser.add_argument("--K", type=int, default=5, help="Number of GMM components")
    parser.add_argument("--T", type=int, default=6, help="Number of refinement iterations")
    parser.add_argument("--init_hidden_dim", type=int, default=256, help="InitNet hidden dim")
    parser.add_argument("--init_num_layers", type=int, default=3, help="InitNet layers")
    parser.add_argument("--refine_hidden_dim", type=int, default=128, help="RefineBlock hidden dim")
    parser.add_argument("--refine_num_layers", type=int, default=2, help="RefineBlock layers")
    parser.add_argument("--sigma_min", type=float, default=1e-3, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=5.0, help="Maximum sigma")
    parser.add_argument("--pi_min", type=float, default=0.0, help="Minimum mixing weight (0=no constraint)")
    parser.add_argument("--corr_scale", type=float, default=0.5, help="Scale for bounded corrections")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no_share_weights", action="store_true",
                        help="Use separate RefineBlock weights for each iteration")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lambda_mom", type=float, default=0.0, help="Moment loss weight")
    parser.add_argument("--lambda_pdf", type=float, default=0.0,
                        help="PDF L2 loss weight (0=CE only, 0.5=50%% CE + 50%% PDF, 1=PDF only)")
    
    # Lambda PDF curriculum
    parser.add_argument("--lambda_pdf_curriculum", action="store_true",
                        help="Use curriculum for lambda_pdf (start at 0, ramp up)")
    parser.add_argument("--lambda_pdf_start_epoch", type=int, default=5,
                        help="Epoch to start ramping lambda_pdf")
    parser.add_argument("--lambda_pdf_end_epoch", type=int, default=15,
                        help="Epoch to finish ramping lambda_pdf")
    parser.add_argument("--lambda_pdf_max", type=float, default=0.1,
                        help="Maximum lambda_pdf value for curriculum")
    
    parser.add_argument("--eta_schedule", type=str, default="linear",
                        choices=["uniform", "linear", "final_only"],
                        help="Deep supervision weight schedule")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Scheduler
    parser.add_argument("--no_scheduler", action="store_true", help="Disable LR scheduler")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["cosine", "step"], help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # EMA
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    result = train_lamf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        K=args.K,
        T=args.T,
        init_hidden_dim=args.init_hidden_dim,
        init_num_layers=args.init_num_layers,
        refine_hidden_dim=args.refine_hidden_dim,
        refine_num_layers=args.refine_num_layers,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        pi_min=args.pi_min,
        corr_scale=args.corr_scale,
        dropout=args.dropout,
        share_refine_weights=not args.no_share_weights,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        lambda_mom=args.lambda_mom,
        lambda_pdf=args.lambda_pdf,
        lambda_pdf_curriculum=args.lambda_pdf_curriculum,
        lambda_pdf_start_epoch=args.lambda_pdf_start_epoch,
        lambda_pdf_end_epoch=args.lambda_pdf_end_epoch,
        lambda_pdf_max=args.lambda_pdf_max,
        eta_schedule=args.eta_schedule,
        grad_clip=args.grad_clip,
        use_scheduler=not args.no_scheduler,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved at epoch {result['best_epoch']}")
    print(f"Test metrics: {result['test_metrics']}")


if __name__ == "__main__":
    main()

