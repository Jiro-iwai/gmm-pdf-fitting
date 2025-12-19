"""Training script for MDN model."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.ml_init.model import MDNModel, log_gmm_pdf
from src.ml_init.metrics import (
    compute_pdf_linf_error,
    compute_cdf_linf_error,
    compute_quantile_error,
    compute_cross_entropy,
)


class PDFDataset(Dataset):
    """Dataset for PDF values."""
    
    def __init__(self, z: np.ndarray, f: np.ndarray):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        z : np.ndarray
            Grid points, shape (N,)
        f : np.ndarray
            PDF values, shape (n_samples, N)
        """
        self.z = torch.from_numpy(z).float()
        self.f = torch.from_numpy(f).float()
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, idx):
        return self.z, self.f[idx]


def load_dataset(
    data_dir: Path,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Load dataset from .npz files.
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing train.npz, val.npz, test.npz
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for DataLoader
    
    Returns:
    --------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    z : np.ndarray
        Grid points
    """
    # Load data
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    test_data = np.load(data_dir / "test.npz")
    
    z = train_data["z"]
    f_train = train_data["f"]
    f_val = val_data["f"]
    f_test = test_data["f"]
    
    # Create datasets
    train_dataset = PDFDataset(z, f_train)
    val_dataset = PDFDataset(z, f_val)
    test_dataset = PDFDataset(z, f_test)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader, z


def compute_loss(
    z: torch.Tensor,
    f_true: torch.Tensor,
    alpha: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
    sigma_min: float,
    lambda_mom: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute loss (cross-entropy + optional moment penalty).
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N)
    alpha : torch.Tensor
        Mixing weight logits, shape (batch_size, K)
    mu : torch.Tensor
        Component means, shape (batch_size, K)
    beta : torch.Tensor
        Variance parameter logits, shape (batch_size, K)
    sigma_min : float
        Minimum standard deviation
    lambda_mom : float
        Moment penalty coefficient
    
    Returns:
    --------
    loss : torch.Tensor
        Total loss
    info : dict
        Loss components
    """
    batch_size, N = f_true.shape
    K = alpha.shape[1]
    
    # Convert to GMM parameters
    pi = torch.softmax(alpha, dim=-1)
    sigma = torch.nn.functional.softplus(beta) + sigma_min
    
    # Compute log PDF
    log_f_hat = log_gmm_pdf(z, pi, mu, sigma)  # (batch_size, N)
    
    # Cross-entropy loss
    # L_CE = -∑ f_true * log(f_hat) * w
    w = torch.ones(N, device=z.device) * (z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2  # Trapezoidal rule
    
    f_hat = torch.exp(log_f_hat)
    f_hat_safe = torch.clamp(f_hat, min=1e-12)
    
    ce_loss = -torch.sum(f_true * torch.log(f_hat_safe) * w.unsqueeze(0), dim=1)
    ce_loss = ce_loss.mean()
    
    # Moment penalty (optional)
    mom_loss = torch.tensor(0.0, device=z.device)
    if lambda_mom > 0:
        # Compute raw moments
        moments_true = []
        moments_hat = []
        for n in range(1, 5):
            m_true = torch.sum(f_true * (z.unsqueeze(0) ** n) * w.unsqueeze(0), dim=1)
            m_hat = torch.sum(f_hat * (z.unsqueeze(0) ** n) * w.unsqueeze(0), dim=1)
            moments_true.append(m_true)
            moments_hat.append(m_hat)
        
        # Penalty: sum of squared differences
        mom_loss = sum(
            ((m_hat - m_true) ** 2).mean() for m_true, m_hat in zip(moments_true, moments_hat)
        )
    
    total_loss = ce_loss + lambda_mom * mom_loss
    
    info = {
        "ce_loss": ce_loss.item(),
        "mom_loss": mom_loss.item() if lambda_mom > 0 else 0.0,
        "total_loss": total_loss.item(),
    }
    
    return total_loss, info


def train_mdn_model(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 20,
    lambda_mom: float = 0.0,
    N: int = 64,
    K: int = 5,
    H: int = 128,
    sigma_min: float = 1e-3,
    num_workers: int = 0,
    max_grad_norm: float = 1.0,
    val_subset_size: int = 1024,
    # New optimizer options
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
    scheduler_type: str = "none",
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    early_stopping_patience: int = 0,
    warmup_epochs: int = 0,
    # Model architecture options
    num_layers: int = 2,
    dropout: float = 0.0,
) -> None:
    """
    Train MDN model.
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing train.npz, val.npz, test.npz
    output_dir : Path
        Output directory for checkpoints
    batch_size : int
        Batch size
    lr : float
        Learning rate
    epochs : int
        Number of epochs
    lambda_mom : float
        Moment penalty coefficient
    N : int
        Input dimension
    K : int
        Number of GMM components
    H : int
        Hidden layer dimension
    sigma_min : float
        Minimum standard deviation
    num_workers : int
        Number of workers for DataLoader
    max_grad_norm : float
        Maximum gradient norm for clipping
    val_subset_size : int
        Size of validation subset for detailed metrics
    optimizer_type : str
        Optimizer type: "adam", "adamw", "sgd"
    weight_decay : float
        L2 regularization coefficient
    scheduler_type : str
        Learning rate scheduler: "none", "cosine", "step", "plateau"
    scheduler_step_size : int
        Step size for StepLR scheduler
    scheduler_gamma : float
        Decay rate for StepLR scheduler
    early_stopping_patience : int
        Early stopping patience (0 = disabled)
    warmup_epochs : int
        Number of warmup epochs (0 = disabled)
    num_layers : int
        Number of hidden layers (2 or 3)
    dropout : float
        Dropout probability (0.0 = disabled)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, test_loader, z = load_dataset(
        data_dir, batch_size, num_workers
    )
    z_torch = torch.from_numpy(z).float()
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDNModel(N=N, K=K, H=H, sigma_min=sigma_min, num_layers=num_layers, dropout=dropout).to(device)
    z_torch = z_torch.to(device)
    
    # Optimizer
    optimizer_type_lower = optimizer_type.lower()
    if optimizer_type_lower == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type_lower == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type_lower == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Learning rate scheduler
    scheduler = None
    scheduler_type_lower = scheduler_type.lower()
    if scheduler_type_lower == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type_lower == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type_lower == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif scheduler_type_lower != "none":
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Training loop
    best_val_ce = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    base_lr = lr
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "pdf_linf": [],
        "cdf_linf": [],
        "epoch_time": [],
        "lr": [],
    }
    
    # Prepare validation subset (first val_subset_size samples)
    val_subset = []
    val_iter = iter(val_loader)
    collected = 0
    while collected < val_subset_size:
        try:
            z_batch, f_batch = next(val_iter)
            val_subset.append((z_batch[0].to(device), f_batch.to(device)))
            collected += len(f_batch)
            if collected >= val_subset_size:
                break
        except StopIteration:
            break
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Warmup learning rate
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training
        model.train()
        train_losses = []
        for z_batch, f_batch in train_loader:
            z_batch = z_batch[0].to(device)  # All samples share same z
            f_batch = f_batch.to(device)
            
            optimizer.zero_grad()
            alpha, mu, beta = model(f_batch)
            loss, loss_info = compute_loss(
                z_batch, f_batch, alpha, mu, beta, sigma_min, lambda_mom
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            train_losses.append(loss_info["ce_loss"])
        
        train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for z_batch, f_batch in val_loader:
                z_batch = z_batch[0].to(device)
                f_batch = f_batch.to(device)
                
                alpha, mu, beta = model(f_batch)
                loss, loss_info = compute_loss(
                    z_batch, f_batch, alpha, mu, beta, sigma_min, lambda_mom
                )
                val_losses.append(loss_info["ce_loss"])
        
        val_loss = np.mean(val_losses)
        
        # Detailed metrics on subset (optional, every epoch)
        val_metrics = {}
        if len(val_subset) > 0:
            with torch.no_grad():
                z_subset = val_subset[0][0]
                f_subset_list = [f for _, f in val_subset[:min(16, len(val_subset))]]
                
                if len(f_subset_list) > 0:
                    f_subset = torch.cat(f_subset_list, dim=0)
                    alpha_subset, mu_subset, beta_subset = model(f_subset)
                    pi_subset = torch.softmax(alpha_subset, dim=-1)
                    sigma_subset = torch.nn.functional.softplus(beta_subset) + sigma_min
                    
                    # Compute metrics for first sample
                    f_true_np = f_subset[0].cpu().numpy()
                    log_f_hat = log_gmm_pdf(z_subset, pi_subset[0:1], mu_subset[0:1], sigma_subset[0:1])
                    f_hat_np = torch.exp(log_f_hat[0]).cpu().numpy()
                    
                    val_metrics["pdf_linf"] = compute_pdf_linf_error(z, f_true_np, f_hat_np)
                    val_metrics["cdf_linf"] = compute_cdf_linf_error(z, f_true_np, f_hat_np)
        
        epoch_time = time.time() - epoch_start_time
        
        # Record history
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["pdf_linf"].append(float(val_metrics.get("pdf_linf", 0)))
        history["cdf_linf"].append(float(val_metrics.get("cdf_linf", 0)))
        history["epoch_time"].append(float(epoch_time))
        history["lr"].append(float(current_lr))
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"lr={current_lr:.6f}, time={epoch_time:.2f}s")
        if val_metrics:
            print(f"  val_metrics: PDF L∞={val_metrics.get('pdf_linf', 0):.6f}, "
                  f"CDF L∞={val_metrics.get('cdf_linf', 0):.6f}")
        
        # Save best model
        if val_loss < best_val_ce:
            best_val_ce = val_loss
            best_epoch = epoch
            
            checkpoint_path = output_dir / f"mdn_init_v1_N{N}_K{K}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save metadata
            metadata = {
                "version": "mdn_init_v1",
                "N_model": N,
                "K_model": K,
                "z_min": float(z[0]),
                "z_max": float(z[-1]),
                "sigma_min": sigma_min,
                "reg_var": sigma_min ** 2,
                "input_transform": "pdf",
                "train_args": {
                    "batch_size": batch_size,
                    "lr": lr,
                    "epochs": epochs,
                    "lambda_mom": lambda_mom,
                    "H": H,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "optimizer": optimizer_type,
                    "weight_decay": weight_decay,
                    "scheduler": scheduler_type,
                    "early_stopping_patience": early_stopping_patience,
                    "warmup_epochs": warmup_epochs,
                },
                "best_epoch": best_epoch,
                "best_val_ce": float(best_val_ce),
                "created_at": datetime.now().isoformat(),
            }
            
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Update scheduler (after warmup)
        if scheduler is not None and epoch >= warmup_epochs:
            if scheduler_type_lower == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
    
    # Save training history
    history["best_epoch"] = best_epoch
    history["best_val_ce"] = float(best_val_ce)
    history_path = output_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best model at epoch {best_epoch+1} with val_loss={best_val_ce:.6f}")
    print(f"Training history saved to {history_path}")


def plot_training_history(history_path: Path, output_path: Path | None = None) -> None:
    """
    Plot training history from saved JSON file.
    
    Parameters:
    -----------
    history_path : Path
        Path to history.json file
    output_path : Path, optional
        Path to save the plot. If None, displays the plot.
    """
    import matplotlib.pyplot as plt
    
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", -1) + 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue", alpha=0.7)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="orange", alpha=0.7)
    if best_epoch > 0:
        ax1.axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PDF L∞ error
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["pdf_linf"], label="PDF L∞", color="red", alpha=0.7)
    if best_epoch > 0:
        ax2.axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PDF L∞ Error")
    ax2.set_title("PDF L∞ Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CDF L∞ error
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["cdf_linf"], label="CDF L∞", color="purple", alpha=0.7)
    if best_epoch > 0:
        ax3.axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("CDF L∞ Error")
    ax3.set_title("CDF L∞ Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epoch time
    ax4 = axes[1, 1]
    ax4.plot(epochs, history["epoch_time"], label="Epoch Time", color="gray", alpha=0.7)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Epoch Training Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train MDN model")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train MDN model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    train_parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    train_parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--lambda_mom", type=float, default=0.0, help="Moment penalty coefficient")
    train_parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size (H)")
    # New optimizer options
    train_parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer type")
    train_parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    train_parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step", "plateau"], help="LR scheduler")
    train_parser.add_argument("--early_stopping", type=int, default=0, help="Early stopping patience (0=disabled)")
    train_parser.add_argument("--warmup_epochs", type=int, default=0, help="Warmup epochs")
    # Model architecture options
    train_parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers")
    train_parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot training history")
    plot_parser.add_argument("--history_path", type=str, required=True, help="Path to history.json")
    plot_parser.add_argument("--output_path", type=str, default=None, help="Path to save plot (optional)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_mdn_model(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            lambda_mom=args.lambda_mom,
            H=args.hidden_size,
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            early_stopping_patience=args.early_stopping,
            warmup_epochs=args.warmup_epochs,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.command == "plot":
        plot_training_history(
            history_path=Path(args.history_path),
            output_path=Path(args.output_path) if args.output_path else None,
        )
    else:
        # Backward compatibility: if no command, assume train with old args
        parser_old = argparse.ArgumentParser(description="Train MDN model")
        parser_old.add_argument("--data_dir", type=str, required=True, help="Data directory")
        parser_old.add_argument("--output_dir", type=str, required=True, help="Output directory")
        parser_old.add_argument("--batch_size", type=int, default=256, help="Batch size")
        parser_old.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        parser_old.add_argument("--epochs", type=int, default=20, help="Number of epochs")
        parser_old.add_argument("--lambda_mom", type=float, default=0.0, help="Moment penalty coefficient")
        parser_old.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size (H)")
        
        args_old = parser_old.parse_args()
        train_mdn_model(
            data_dir=Path(args_old.data_dir),
            output_dir=Path(args_old.output_dir),
            batch_size=args_old.batch_size,
            lr=args_old.lr,
            epochs=args_old.epochs,
            lambda_mom=args_old.lambda_mom,
            H=args_old.hidden_size,
        )


if __name__ == "__main__":
    main()

