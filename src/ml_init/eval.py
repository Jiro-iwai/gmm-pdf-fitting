"""Evaluation script for MDN model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.ml_init.model import MDNModel, log_gmm_pdf
from src.ml_init.metrics import (
    compute_pdf_linf_error,
    compute_cdf_linf_error,
    compute_quantile_error,
    compute_cross_entropy,
)


def load_model_and_metadata(
    model_path: Path,
) -> tuple[MDNModel, dict, np.ndarray]:
    """
    Load trained model and metadata.
    
    Parameters:
    -----------
    model_path : Path
        Path to model checkpoint (.pt file)
    
    Returns:
    --------
    model : MDNModel
        Loaded model
    metadata : dict
        Model metadata
    z : np.ndarray
        Grid points
    """
    model_path = Path(model_path)
    
    # Handle both directory and file paths
    if model_path.is_dir():
        # Directory passed: look for .pt file and metadata.json inside
        pt_files = list(model_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in directory: {model_path}")
        model_pt_path = pt_files[0]  # Use first .pt file
        metadata_path = model_path / "metadata.json"
    else:
        # File passed: metadata is in same directory
        model_pt_path = model_path
        metadata_path = model_path.parent / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Extract parameters
    N = metadata["N_model"]
    K = metadata["K_model"]
    sigma_min = metadata["sigma_min"]
    z_min = metadata["z_min"]
    z_max = metadata["z_max"]
    
    # Get model architecture from metadata (with backward compatibility defaults)
    train_args = metadata.get("train_args", {})
    H = train_args.get("H", 128)
    num_layers = train_args.get("num_layers", 2)
    dropout = train_args.get("dropout", 0.0)
    use_layernorm = train_args.get("use_layernorm", False)
    use_residual = train_args.get("use_residual", False)
    
    # Create grid
    z = np.linspace(z_min, z_max, N)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDNModel(
        N=N, K=K, H=H, sigma_min=sigma_min, 
        num_layers=num_layers, dropout=dropout,
        use_layernorm=use_layernorm, use_residual=use_residual
    ).to(device)
    
    # Load state dict with backward compatibility
    state_dict = torch.load(model_pt_path, map_location=device, weights_only=True)
    
    # Check if old format (fc1, fc2, fc3) and convert to new format
    if "fc1.weight" in state_dict and "hidden.0.weight" not in state_dict:
        # Old format: convert to new
        new_state_dict = {}
        new_state_dict["hidden.0.weight"] = state_dict["fc1.weight"]
        new_state_dict["hidden.0.bias"] = state_dict["fc1.bias"]
        # Skip ReLU (index 1) and Dropout (index 2 if present)
        layer_idx = 3 if dropout > 0 else 2
        new_state_dict[f"hidden.{layer_idx}.weight"] = state_dict["fc2.weight"]
        new_state_dict[f"hidden.{layer_idx}.bias"] = state_dict["fc2.bias"]
        new_state_dict["fc_out.weight"] = state_dict["fc3.weight"]
        new_state_dict["fc_out.bias"] = state_dict["fc3.bias"]
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, metadata, z


def evaluate_mdn_model(
    model_path: Path,
    test_data_path: Path,
    output_path: Path | None = None,
    quantiles: list[float] | None = None,
) -> dict:
    """
    Evaluate MDN model on test dataset.
    
    Parameters:
    -----------
    model_path : Path
        Path to model checkpoint
    test_data_path : Path
        Path to test data (.npz file)
    output_path : Path, optional
        Path to save results JSON
    quantiles : list[float], optional
        Quantile levels for evaluation (default: [0.5, 0.9, 0.99])
    
    Returns:
    --------
    results : dict
        Evaluation results
    """
    if quantiles is None:
        quantiles = [0.5, 0.9, 0.99]
    
    # Load model
    model, metadata, z = load_model_and_metadata(model_path)
    device = next(model.parameters()).device
    z_torch = torch.from_numpy(z).float().to(device)
    
    # Load test data
    test_data = np.load(test_data_path)
    f_test = test_data["f"]
    n_samples = len(f_test)
    
    # Evaluation metrics
    ce_list = []
    pdf_linf_list = []
    cdf_linf_list = []
    quantile_errors_list = {q: [] for q in quantiles}
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            f_true = f_test[i]
            f_true_torch = torch.from_numpy(f_true).float().unsqueeze(0).to(device)
            
            # Forward pass
            alpha, mu, beta = model(f_true_torch)
            pi = torch.softmax(alpha, dim=-1)
            sigma = torch.nn.functional.softplus(beta) + model.sigma_min
            
            # Compute log PDF
            log_f_hat = log_gmm_pdf(z_torch, pi, mu, sigma)
            f_hat = torch.exp(log_f_hat[0]).cpu().numpy()
            
            # Compute metrics
            ce = compute_cross_entropy(z, f_true, f_hat)
            pdf_linf = compute_pdf_linf_error(z, f_true, f_hat)
            cdf_linf = compute_cdf_linf_error(z, f_true, f_hat)
            q_errors = compute_quantile_error(z, f_true, f_hat, quantiles)
            
            ce_list.append(ce)
            pdf_linf_list.append(pdf_linf)
            cdf_linf_list.append(cdf_linf)
            for q, err in zip(quantiles, q_errors):
                quantile_errors_list[q].append(err)
    
    # Aggregate results
    results = {
        "n_samples": n_samples,
        "mean_ce": float(np.mean(ce_list)),
        "std_ce": float(np.std(ce_list)),
        "mean_pdf_linf": float(np.mean(pdf_linf_list)),
        "std_pdf_linf": float(np.std(pdf_linf_list)),
        "mean_cdf_linf": float(np.mean(cdf_linf_list)),
        "std_cdf_linf": float(np.std(cdf_linf_list)),
        "quantile_errors": {
            q: {
                "mean": float(np.mean(quantile_errors_list[q])),
                "std": float(np.std(quantile_errors_list[q])),
            }
            for q in quantiles
        },
    }
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate MDN model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, help="Test data path (.npz)")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    results = evaluate_mdn_model(
        model_path=Path(args.model_path),
        test_data_path=Path(args.data_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    
    # Print summary
    print(f"Evaluation Results (n={results['n_samples']}):")
    print(f"  Mean CE: {results['mean_ce']:.6f} ± {results['std_ce']:.6f}")
    print(f"  Mean PDF L∞: {results['mean_pdf_linf']:.6f} ± {results['std_pdf_linf']:.6f}")
    print(f"  Mean CDF L∞: {results['mean_cdf_linf']:.6f} ± {results['std_cdf_linf']:.6f}")
    print("  Quantile Errors:")
    for q, q_data in results['quantile_errors'].items():
        print(f"    {q}: {q_data['mean']:.6f} ± {q_data['std']:.6f}")


if __name__ == "__main__":
    main()

