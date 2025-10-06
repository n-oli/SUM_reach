

"""
evaluate_robustness.py

This script orchestrates the final, comprehensive evaluation of all trained models
to determine the most robust defense strategy.

It performs the following steps:
1.  Automatically discovers all trained models (`best_model.pth`) in a given
    directory (e.g., 'Neural_nets/models/').
2.  Includes the original, non-adversarially trained model as a baseline.
3.  For each discovered model, it runs a gauntlet of evaluations:
    a. Accuracy on the clean, un-attacked test set.
    b. Robustness against the FGSM attack at various epsilon strengths.
    c. Robustness against the stronger PGD attack at various epsilon strengths.
4.  Collects all results and generates a final, human-readable summary report
    in Markdown format, which is both printed to the console and saved to
    `evaluation_summary.md`.
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader

# --- CleverHans Imports ---
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# Import custom modules
from data_loader import MalwareDataset
from model import MalwareDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_clean(model, test_loader, device):
    """Evaluates model accuracy on a clean dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (api_batch, global_batch), labels in test_loader:
            api_batch, global_batch, labels = api_batch.to(device), global_batch.to(device), labels.to(device)
            outputs = model(api_batch, global_batch)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def evaluate_adversarial(model, test_loader, attack_fn, attack_kwargs, device):
    """Evaluates model accuracy on an adversarial dataset."""
    model.eval()
    correct = 0
    total = 0
    for (api_batch, global_batch), labels in test_loader:
        api_batch, global_batch, labels = api_batch.to(device), global_batch.to(device), labels.to(device)
        
        def model_fn(x):
            probs_malware = model(x, global_batch)
            probs_benign = 1.0 - probs_malware
            return torch.stack([probs_benign, probs_malware], dim=-1)

        batch_attack_kwargs = attack_kwargs.copy()
        batch_attack_kwargs['y'] = labels.long()

        model.train() # Switch to train mode for gradient calculation
        api_batch_adv = attack_fn(model_fn, api_batch, **batch_attack_kwargs)
        model.eval() # Switch back to eval mode for prediction

        with torch.no_grad():
            outputs = model(api_batch_adv, global_batch)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Test Data ---
    splits_path = os.path.join(args.data_dir, 'dataset_splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    test_dataset = MalwareDataset(splits['test']['files'], splits['test']['labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    logging.info(f"Test set loaded with {len(test_dataset)} samples.")

    # --- Discover Models ---
    model_paths = [os.path.join(args.data_dir, 'best_model.pth')] # Baseline model
    model_paths.extend(glob(os.path.join(args.models_dir, '**/best_model.pth'), recursive=True))
    logging.info(f"Found {len(model_paths)} models to evaluate.")

    # --- Define Evaluation Gauntlet ---
    epsilons = [0.01, 0.03, 0.06, 0.08, 0.1]
    results_data = []

    # --- Run Evaluations ---
    for path in model_paths:
        model_name = os.path.basename(os.path.dirname(path))
        if 'Neural_nets' in model_name: model_name = "Baseline (Clean)"
        
        logging.info(f"--- Evaluating model: {model_name} ---")
        
        model = MalwareDetectionModel().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        
        model_results = {'Model': model_name}

        # 1. Clean Accuracy
        clean_acc = evaluate_clean(model, test_loader, device)
        model_results['Clean Acc.'] = f"{clean_acc:.2%}"
        logging.info(f"  Clean Accuracy: {clean_acc:.4f}")

        # 2. FGSM Attacks
        for eps in epsilons:
            logging.info(f"  Attacking with FGSM (eps={eps})...")
            kwargs = {'eps': eps, 'norm': np.inf}
            fgsm_acc = evaluate_adversarial(model, test_loader, fast_gradient_method, kwargs, device)
            model_results[f'FGSM (ε={eps})'] = f"{fgsm_acc:.2%}"
        
        # 3. PGD Attacks
        for eps in epsilons:
            logging.info(f"  Attacking with PGD (eps={eps})...")
            kwargs = {'eps': eps, 'eps_iter': (2.5 * eps) / 40, 'nb_iter': 40, 'norm': np.inf}
            pgd_acc = evaluate_adversarial(model, test_loader, projected_gradient_descent, kwargs, device)
            model_results[f'PGD (ε={eps})'] = f"{pgd_acc:.2%}"
            
        results_data.append(model_results)

    # --- Generate and Save Report ---
    df = pd.DataFrame(results_data)
    df = df.set_index('Model')
    
    # Reorder columns for clarity
    cols = ['Clean Acc.'] + sorted([c for c in df.columns if c != 'Clean Acc.'], key=lambda x: (x.split(' ')[0], float(x.split('=')[1][:-1])))
    df = df[cols]

    report_md = df.to_markdown()
    
    print("\n\n--- Comprehensive Robustness Evaluation Report ---")
    print(report_md)
    
    report_path = os.path.join(args.output_dir, 'evaluation_summary.md')
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Robustness Evaluation Report\n\n")
        f.write(report_md)
    logging.info(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the robustness of all trained models.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_dir', type=str, default=script_dir, help="Directory with dataset splits.")
    parser.add_argument('--models_dir', type=str, default=os.path.join(script_dir, 'models'), help="Directory containing subdirectories of trained models.")
    parser.add_argument('--output_dir', type=str, default=script_dir, help="Directory to save the final report.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA.")

    args = parser.parse_args()
    main(args)
