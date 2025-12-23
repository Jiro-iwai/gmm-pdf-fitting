"""Evaluation script for LAMF model.

This module provides comprehensive evaluation utilities for LAMF models,
including comparison with EM baseline.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from .model import LAMFFitter
from .dataset import LAMFDataset
from .metrics import compute_gmm_pdf, evaluate_gmm_fit


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[LAMFFitter, dict]:
    """
    Load LAMF model from checkpoint.
    
    Parameters:
    -----------
    model_path : str
        Path to model directory or checkpoint file
    device : torch.device
        Device to load model on
    
    Returns:
    --------
    model : LAMFFitter
        Loaded model
    metadata : dict
        Model metadata
    """
    model_path = Path(model_path)
    
    if model_path.is_dir():
        checkpoint_path = model_path / "lamf_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_path / "best_model.pt"
        metadata_path = model_path / "metadata.json"
    else:
        checkpoint_path = model_path
        metadata_path = model_path.parent / "metadata.json"
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = LAMFFitter(
        N=metadata['N'],
        K=metadata['K'],
        T=metadata['T'],
        init_hidden_dim=metadata.get('init_hidden_dim', 256),
        init_num_layers=metadata.get('init_num_layers', 3),
        refine_hidden_dim=metadata.get('refine_hidden_dim', 128),
        refine_num_layers=metadata.get('refine_num_layers', 2),
        sigma_min=metadata.get('sigma_min', 1e-3),
        dropout=metadata.get('dropout', 0.1),
        share_refine_weights=metadata.get('share_refine_weights', True),
    )
    
    # Load weights
    if str(checkpoint_path).endswith('best_model.pt'):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model, metadata


@torch.no_grad()
def evaluate_on_dataset(
    model: LAMFFitter,
    dataset: LAMFDataset,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Evaluate LAMF model on a dataset.
    
    Parameters:
    -----------
    model : LAMFFitter
        LAMF model
    dataset : LAMFDataset
        Dataset to evaluate on
    device : torch.device
        Device
    max_samples : int, optional
        Maximum samples to evaluate (None = all)
    
    Returns:
    --------
    results : dict
        Evaluation results
    """
    model.eval()
    
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    ce_list = []
    pdf_linf_list = []
    cdf_linf_list = []
    m1_error_list = []
    m2_error_list = []
    inference_times = []
    
    z_tensor = dataset.get_z_tensor().to(device)
    
    for i in range(n_samples):
        sample = dataset[i]
        w = sample['w'].unsqueeze(0).to(device)
        f_true = sample['f'].numpy()
        z_np = sample['z'].numpy()
        
        # Time inference
        start_time = time.perf_counter()
        result = model(z_tensor, w)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        inference_time = time.perf_counter() - start_time
        
        inference_times.append(inference_time)
        
        # Extract parameters
        pi = result['pi'][0].cpu().numpy()
        mu = result['mu'][0].cpu().numpy()
        sigma = result['sigma'][0].cpu().numpy()
        
        # Evaluate
        metrics = evaluate_gmm_fit(z_np, f_true, pi, mu, sigma)
        
        ce_list.append(metrics['cross_entropy'])
        pdf_linf_list.append(metrics['pdf_linf'])
        cdf_linf_list.append(metrics['cdf_linf'])
        m1_error_list.append(metrics['M1_error'])
        m2_error_list.append(metrics['M2_error'])
    
    return {
        'n_samples': n_samples,
        'ce_mean': float(np.mean(ce_list)),
        'ce_std': float(np.std(ce_list)),
        'pdf_linf_mean': float(np.mean(pdf_linf_list)),
        'pdf_linf_std': float(np.std(pdf_linf_list)),
        'pdf_linf_max': float(np.max(pdf_linf_list)),
        'pdf_linf_p95': float(np.percentile(pdf_linf_list, 95)),
        'cdf_linf_mean': float(np.mean(cdf_linf_list)),
        'cdf_linf_std': float(np.std(cdf_linf_list)),
        'cdf_linf_max': float(np.max(cdf_linf_list)),
        'm1_error_mean': float(np.mean(m1_error_list)),
        'm2_error_mean': float(np.mean(m2_error_list)),
        'inference_time_mean': float(np.mean(inference_times)),
        'inference_time_std': float(np.std(inference_times)),
        'inference_time_total': float(np.sum(inference_times)),
    }


def compare_with_em(
    model: LAMFFitter,
    dataset: LAMFDataset,
    device: torch.device,
    n_samples: int = 100,
    em_kwargs: Optional[dict] = None,
) -> dict:
    """
    Compare LAMF with EM method on the same samples.
    
    Parameters:
    -----------
    model : LAMFFitter
        LAMF model
    dataset : LAMFDataset
        Dataset to evaluate on
    device : torch.device
        Device
    n_samples : int
        Number of samples to compare
    em_kwargs : dict, optional
        Additional kwargs for EM method
    
    Returns:
    --------
    comparison : dict
        Comparison results
    """
    from src.gmm_fitting.em_method import fit_gmm1d_to_pdf_weighted_em as fit_gmm_em
    
    model.eval()
    z_tensor = dataset.get_z_tensor().to(device)
    
    lamf_results = []
    em_results = []
    
    em_kwargs = em_kwargs or {}
    default_em_kwargs = {
        'init': 'quantile',
        'n_init': 5,
        'max_iter': 300,
        'tol': 1e-6,
    }
    em_kwargs = {**default_em_kwargs, **em_kwargs}
    
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        w = sample['w'].unsqueeze(0).to(device)
        f_true = sample['f'].numpy()
        z_np = sample['z'].numpy()
        
        # LAMF inference
        lamf_start = time.perf_counter()
        with torch.no_grad():
            result = model(z_tensor, w)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        lamf_time = time.perf_counter() - lamf_start
        
        lamf_pi = result['pi'][0].cpu().numpy()
        lamf_mu = result['mu'][0].cpu().numpy()
        lamf_sigma = result['sigma'][0].cpu().numpy()
        
        lamf_metrics = evaluate_gmm_fit(z_np, f_true, lamf_pi, lamf_mu, lamf_sigma)
        lamf_metrics['time'] = lamf_time
        lamf_results.append(lamf_metrics)
        
        # EM inference
        em_start = time.perf_counter()
        try:
            gmm_params, log_likelihood, iterations = fit_gmm_em(
                z=z_np,
                f=f_true,
                K=model.K,
                **em_kwargs,
            )
            em_time = time.perf_counter() - em_start
            
            em_pi = gmm_params.pi
            em_mu = gmm_params.mu
            em_sigma = np.sqrt(gmm_params.var)
            
            em_metrics = evaluate_gmm_fit(z_np, f_true, em_pi, em_mu, em_sigma)
            em_metrics['time'] = em_time
            em_metrics['iterations'] = iterations
            em_metrics['success'] = True
        except Exception as e:
            em_metrics = {
                'pdf_linf': float('inf'),
                'cdf_linf': float('inf'),
                'cross_entropy': float('inf'),
                'time': time.perf_counter() - em_start,
                'iterations': -1,
                'success': False,
                'error': str(e),
            }
        
        em_results.append(em_metrics)
    
    # Aggregate results
    lamf_pdf_linf = [r['pdf_linf'] for r in lamf_results]
    lamf_cdf_linf = [r['cdf_linf'] for r in lamf_results]
    lamf_times = [r['time'] for r in lamf_results]
    
    em_pdf_linf = [r['pdf_linf'] for r in em_results if r['success']]
    em_cdf_linf = [r['cdf_linf'] for r in em_results if r['success']]
    em_times = [r['time'] for r in em_results if r['success']]
    em_iterations = [r['iterations'] for r in em_results if r['success']]
    em_success_rate = sum(1 for r in em_results if r['success']) / len(em_results)
    
    return {
        'n_samples': n_samples,
        'lamf': {
            'pdf_linf_mean': float(np.mean(lamf_pdf_linf)),
            'pdf_linf_std': float(np.std(lamf_pdf_linf)),
            'cdf_linf_mean': float(np.mean(lamf_cdf_linf)),
            'cdf_linf_std': float(np.std(lamf_cdf_linf)),
            'time_mean': float(np.mean(lamf_times)),
            'time_total': float(np.sum(lamf_times)),
            'iterations': model.T,
        },
        'em': {
            'pdf_linf_mean': float(np.mean(em_pdf_linf)) if em_pdf_linf else float('inf'),
            'pdf_linf_std': float(np.std(em_pdf_linf)) if em_pdf_linf else 0,
            'cdf_linf_mean': float(np.mean(em_cdf_linf)) if em_cdf_linf else float('inf'),
            'cdf_linf_std': float(np.std(em_cdf_linf)) if em_cdf_linf else 0,
            'time_mean': float(np.mean(em_times)) if em_times else float('inf'),
            'time_total': float(np.sum(em_times)) if em_times else float('inf'),
            'iterations_mean': float(np.mean(em_iterations)) if em_iterations else -1,
            'success_rate': em_success_rate,
        },
        'speedup': float(np.mean(em_times) / np.mean(lamf_times)) if lamf_times and em_times else 0,
    }


def visualize_samples(
    model: LAMFFitter,
    dataset: LAMFDataset,
    device: torch.device,
    n_samples: int = 6,
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize LAMF predictions on sample PDFs.
    
    Parameters:
    -----------
    model : LAMFFitter
        LAMF model
    dataset : LAMFDataset
        Dataset
    device : torch.device
        Device
    n_samples : int
        Number of samples to visualize
    output_path : str, optional
        Path to save figure
    """
    model.eval()
    z_tensor = dataset.get_z_tensor().to(device)
    
    # Select samples evenly spaced
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, idx in zip(axes, indices):
        sample = dataset[int(idx)]
        w = sample['w'].unsqueeze(0).to(device)
        f_true = sample['f'].numpy()
        z_np = sample['z'].numpy()
        params = sample['params'].numpy()
        
        # LAMF prediction
        with torch.no_grad():
            result = model(z_tensor, w)
        
        pi = result['pi'][0].cpu().numpy()
        mu = result['mu'][0].cpu().numpy()
        sigma = result['sigma'][0].cpu().numpy()
        
        # Compute predicted PDF
        f_hat = np.zeros_like(z_np)
        inv_sqrt_2pi = 1.0 / np.sqrt(2 * np.pi)
        for k in range(len(pi)):
            f_hat += pi[k] * inv_sqrt_2pi / sigma[k] * np.exp(
                -0.5 * ((z_np - mu[k]) / sigma[k]) ** 2
            )
        
        # Plot
        ax.plot(z_np, f_true, 'b-', label='Target', linewidth=2)
        ax.plot(z_np, f_hat, 'r--', label='LAMF', linewidth=2)
        ax.fill_between(z_np, f_true, alpha=0.3)
        
        # Add metrics
        metrics = evaluate_gmm_fit(z_np, f_true, pi, mu, sigma)
        ax.set_title(
            f"Sample {idx}\n"
            f"PDF L∞: {metrics['pdf_linf']:.4f}, CDF L∞: {metrics['cdf_linf']:.4f}"
        )
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LAMF model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LAMF model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--compare_em", action="store_true",
                        help="Compare with EM method")
    parser.add_argument("--n_compare", type=int, default=100,
                        help="Number of samples for EM comparison")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample predictions")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, metadata = load_model(args.model_path, device)
    print(f"Model: N={metadata['N']}, K={metadata['K']}, T={metadata['T']}")
    
    # Load dataset
    data_path = Path(args.data_dir) / f"{args.split}.npz"
    print(f"\nLoading dataset from {data_path}...")
    dataset = LAMFDataset(data_path)
    print(f"Dataset: {len(dataset)} samples")
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    results = evaluate_on_dataset(
        model, dataset, device,
        max_samples=args.max_samples,
    )
    
    print(f"\n{'='*50}")
    print(f"LAMF Evaluation Results ({args.split} set)")
    print(f"{'='*50}")
    print(f"Samples evaluated: {results['n_samples']}")
    print(f"Cross-Entropy:     {results['ce_mean']:.4f} ± {results['ce_std']:.4f}")
    print(f"PDF L∞:            {results['pdf_linf_mean']:.4f} ± {results['pdf_linf_std']:.4f}")
    print(f"PDF L∞ (max):      {results['pdf_linf_max']:.4f}")
    print(f"PDF L∞ (p95):      {results['pdf_linf_p95']:.4f}")
    print(f"CDF L∞:            {results['cdf_linf_mean']:.4f} ± {results['cdf_linf_std']:.4f}")
    print(f"M1 Error:          {results['m1_error_mean']:.4f}")
    print(f"M2 Error:          {results['m2_error_mean']:.4f}")
    print(f"Inference Time:    {results['inference_time_mean']*1000:.2f} ± {results['inference_time_std']*1000:.2f} ms")
    
    # Compare with EM
    if args.compare_em:
        print(f"\n{'='*50}")
        print(f"Comparing with EM method...")
        print(f"{'='*50}")
        
        comparison = compare_with_em(
            model, dataset, device,
            n_samples=args.n_compare,
        )
        
        print(f"\nLAMF vs EM Comparison ({comparison['n_samples']} samples):")
        print(f"\n{'Method':<15} {'PDF L∞':<20} {'CDF L∞':<20} {'Time (ms)':<15}")
        print(f"{'-'*70}")
        print(
            f"{'LAMF':<15} "
            f"{comparison['lamf']['pdf_linf_mean']:.4f} ± {comparison['lamf']['pdf_linf_std']:.4f}   "
            f"{comparison['lamf']['cdf_linf_mean']:.4f} ± {comparison['lamf']['cdf_linf_std']:.4f}   "
            f"{comparison['lamf']['time_mean']*1000:.2f}"
        )
        print(
            f"{'EM':<15} "
            f"{comparison['em']['pdf_linf_mean']:.4f} ± {comparison['em']['pdf_linf_std']:.4f}   "
            f"{comparison['em']['cdf_linf_mean']:.4f} ± {comparison['em']['cdf_linf_std']:.4f}   "
            f"{comparison['em']['time_mean']*1000:.2f}"
        )
        print(f"\nSpeedup: {comparison['speedup']:.1f}x")
        print(f"EM success rate: {comparison['em']['success_rate']*100:.1f}%")
        print(f"EM avg iterations: {comparison['em']['iterations_mean']:.1f}")
        
        results['comparison'] = comparison
    
    # Visualize
    if args.visualize:
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"samples_{args.split}.png")
        
        visualize_samples(model, dataset, device, output_path=output_path)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"eval_results_{args.split}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

