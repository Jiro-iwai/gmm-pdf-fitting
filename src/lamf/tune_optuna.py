#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for LAMF V5-Linf model.

Phase 1: L∞ loss parameter optimization
- lambda_linf_max: Final L∞ loss weight
- linf_alpha: Soft L∞ temperature
- lambda_linf_start_epoch: Epoch to start L∞ curriculum
- lambda_linf_end_epoch: Epoch to finish L∞ curriculum

Usage:
    python -m src.lamf.tune_optuna --n_trials 20 --study_name v5_linf_phase1
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import LAMFDataset
from .model import LAMFFitter
from .metrics import (
    compute_cross_entropy_loss,
    compute_deep_supervision_loss,
    compute_gmm_pdf,
    compute_pdf_linf_error,
    compute_cdf_linf_error,
)


def create_model(config: dict) -> LAMFFitter:
    """Create LAMF model with given config."""
    return LAMFFitter(
        N=config['N'],
        K=config['K'],
        T=config['T'],
        init_hidden_dim=config['init_hidden_dim'],
        init_num_layers=config['init_num_layers'],
        refine_hidden_dim=config['refine_hidden_dim'],
        refine_num_layers=config['refine_num_layers'],
        sigma_min=config['sigma_min'],
        sigma_max=config['sigma_max'],
        pi_min=config.get('pi_min', 0.0),
        corr_scale=config['corr_scale'],
        dropout=config['dropout'],
        share_refine_weights=config.get('share_refine_weights', True),
        use_attention=False,
    )


def train_one_epoch(
    model: LAMFFitter,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_pdf: float,
    lambda_linf: float,
    linf_alpha: float,
    eta_schedule: str = "final_only",
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0
    
    for batch in train_loader:
        z = batch['z'][0].to(device)
        w = batch['w'].to(device)
        f_true = batch['f'].to(device) if lambda_linf > 0 or lambda_pdf > 0 else None
        
        optimizer.zero_grad()
        
        result = model(z, w, return_intermediate=True)
        intermediate = result['intermediate']
        
        loss = compute_deep_supervision_loss(
            z, w, intermediate,
            lambda_mom=0.0,
            lambda_pdf=lambda_pdf,
            lambda_linf=lambda_linf,
            lambda_topk=0.0,
            f_true=f_true,
            eta_schedule=eta_schedule,
            linf_alpha=linf_alpha,
            topk_k=10,
        )
        
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue
        
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        with torch.no_grad():
            final_pi, final_mu, final_sigma = result['pi'], result['mu'], result['sigma']
            ce = compute_cross_entropy_loss(z, w, final_pi, final_mu, final_sigma).mean()
            
            total_loss += loss.item()
            total_ce += ce.item()
        
        n_batches += 1
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'ce': total_ce / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: LAMFFitter,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    
    total_ce = 0.0
    total_pdf_linf = 0.0
    total_cdf_linf = 0.0
    n_samples = 0
    
    for batch in val_loader:
        z = batch['z'][0].to(device)
        w = batch['w'].to(device)
        f_true = batch['f'].to(device)
        
        batch_size = w.shape[0]
        
        result = model(z, w)
        pi, mu, sigma = result['pi'], result['mu'], result['sigma']
        
        ce = compute_cross_entropy_loss(z, w, pi, mu, sigma)
        total_ce += ce.sum().item()
        
        f_hat = compute_gmm_pdf(z, pi, mu, sigma)
        pdf_linf = compute_pdf_linf_error(z, f_true, f_hat)
        cdf_linf = compute_cdf_linf_error(z, f_true, f_hat)
        
        total_pdf_linf += pdf_linf.sum().item()
        total_cdf_linf += cdf_linf.sum().item()
        
        n_samples += batch_size
    
    return {
        'ce': total_ce / n_samples,
        'pdf_linf': total_pdf_linf / n_samples,
        'cdf_linf': total_cdf_linf / n_samples,
    }


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function."""
    
    # Phase 1: L∞ loss parameters
    lambda_linf_max = trial.suggest_float('lambda_linf_max', 0.2, 1.0, step=0.1)
    linf_alpha = trial.suggest_float('linf_alpha', 10.0, 100.0, log=True)
    lambda_linf_start_epoch = trial.suggest_int('lambda_linf_start_epoch', 3, 10)
    lambda_linf_end_epoch = trial.suggest_int('lambda_linf_end_epoch', 15, 35)
    
    # Ensure end > start
    if lambda_linf_end_epoch <= lambda_linf_start_epoch:
        lambda_linf_end_epoch = lambda_linf_start_epoch + 10
    
    # Fixed parameters (from V5-Linf baseline)
    config = {
        'N': 96,
        'K': 5,
        'T': 6,
        'init_hidden_dim': 256,
        'init_num_layers': 3,
        'refine_hidden_dim': 128,
        'refine_num_layers': 2,
        'sigma_min': 0.01,
        'sigma_max': 5.0,
        'pi_min': 0.0,
        'corr_scale': 0.5,
        'dropout': 0.2,
        'share_refine_weights': True,
    }
    
    # Training parameters
    lr = args.lr
    lambda_pdf = args.lambda_pdf
    warmup_epochs = args.warmup_epochs
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_dir = Path(args.data_dir)
    train_dataset = LAMFDataset(data_dir / 'train.npz')
    val_dataset = LAMFDataset(data_dir / 'val.npz')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_pdf_linf = float('inf')
    
    for epoch in range(epochs):
        # Compute current lambda_linf using curriculum
        if epoch < lambda_linf_start_epoch:
            current_lambda_linf = 0.0
        elif epoch >= lambda_linf_end_epoch:
            current_lambda_linf = lambda_linf_max
        else:
            progress = (epoch - lambda_linf_start_epoch) / (lambda_linf_end_epoch - lambda_linf_start_epoch)
            current_lambda_linf = lambda_linf_max * progress
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_pdf=lambda_pdf,
            lambda_linf=current_lambda_linf,
            linf_alpha=linf_alpha,
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Track best
        if val_metrics['pdf_linf'] < best_val_pdf_linf:
            best_val_pdf_linf = val_metrics['pdf_linf']
        
        # Report to Optuna for pruning
        trial.report(val_metrics['pdf_linf'], epoch)
        
        # Pruning check
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping if already very good
        if val_metrics['pdf_linf'] < 0.008:
            break
    
    return best_val_pdf_linf


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for LAMF')
    
    # Optuna settings
    parser.add_argument('--study_name', type=str, default='v5_linf_phase1',
                        help='Optuna study name')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='ml_init/data_v5',
                        help='Data directory')
    
    # Fixed training parameters
    parser.add_argument('--epochs', type=int, default=40,
                        help='Max epochs per trial')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lambda_pdf', type=float, default=0.2,
                        help='PDF L2 loss weight')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='lamf/optuna_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study
    storage = args.storage or f"sqlite:///{output_dir / 'optuna.db'}"
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
            interval_steps=5,
        ),
    )
    
    print(f"=" * 70)
    print(f"Optuna Hyperparameter Tuning - Phase 1: L∞ Loss Parameters")
    print(f"=" * 70)
    print(f"Study name: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"N trials: {args.n_trials}")
    print(f"Max epochs per trial: {args.epochs}")
    print()
    print(f"Search space:")
    print(f"  lambda_linf_max: [0.2, 1.0]")
    print(f"  linf_alpha: [10.0, 100.0] (log)")
    print(f"  lambda_linf_start_epoch: [3, 10]")
    print(f"  lambda_linf_end_epoch: [15, 35]")
    print()
    
    start_time = time.time()
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Results
    print()
    print(f"=" * 70)
    print(f"Optimization Complete")
    print(f"=" * 70)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print()
    
    # Best trial
    best_trial = study.best_trial
    print(f"Best trial:")
    print(f"  Value (PDF L∞): {best_trial.value:.6f}")
    print(f"  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print()
    
    # Save results
    results = {
        'study_name': args.study_name,
        'n_trials': len(study.trials),
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'elapsed_minutes': elapsed / 60,
        'timestamp': datetime.now().isoformat(),
    }
    
    results_file = output_dir / f'{args.study_name}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Top 5 trials
    print()
    print(f"Top 5 trials:")
    print(f"-" * 70)
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value
    )[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        print(f"{i}. PDF L∞={trial.value:.6f}")
        print(f"   lambda_linf_max={trial.params.get('lambda_linf_max', 'N/A'):.2f}, "
              f"linf_alpha={trial.params.get('linf_alpha', 'N/A'):.1f}, "
              f"start={trial.params.get('lambda_linf_start_epoch', 'N/A')}, "
              f"end={trial.params.get('lambda_linf_end_epoch', 'N/A')}")
    
    # Generate training command with best params
    print()
    print(f"=" * 70)
    print(f"Recommended training command with best parameters:")
    print(f"=" * 70)
    bp = best_trial.params
    print(f"""
python -m src.lamf.train \\
  --data_dir ml_init/data_v5 \\
  --output_dir lamf/checkpoints_v5_linf_tuned \\
  --epochs 50 \\
  --batch_size 64 \\
  --lr {args.lr} \\
  --lambda_pdf {args.lambda_pdf} \\
  --lambda_linf 0.1 \\
  --linf_alpha {bp['linf_alpha']:.1f} \\
  --lambda_linf_curriculum \\
  --lambda_linf_start_epoch {bp['lambda_linf_start_epoch']} \\
  --lambda_linf_end_epoch {bp['lambda_linf_end_epoch']} \\
  --lambda_linf_max {bp['lambda_linf_max']:.2f} \\
  --warmup_epochs {args.warmup_epochs} \\
  --scheduler_type cosine \\
  --grad_clip 1.0 \\
  --patience 20
""")


if __name__ == '__main__':
    main()

