

"""
attack_model.py

This script benchmarks the robustness of a trained model against standard
adversarial attacks (FGSM and PGD) using the CleverHans library.
This version is updated to use the modern (v4+) functional API of CleverHans.

It performs the following steps:
1.  Loads a pre-trained model (e.g., 'best_model.pth').
2.  Loads the unseen test set.
3.  Defines a set of epsilon values to test the attack strength.
4.  For each attack type (FGSM, PGD) and each epsilon value, it:
    a. Calls the attack function from the CleverHans library.
    b. Generates adversarial examples from the clean test set images on-the-fly.
    c. Evaluates the model's accuracy on these adversarial examples.
5.  Prints a summary report showing the model's accuracy under each attack scenario.

This script is crucial for establishing a "vulnerability baseline" before
proceeding with adversarial training.
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

# --- CleverHans Imports (Modern Functional API) ---
# Ensure you have installed cleverhans: pip install cleverhans
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# Import custom modules
from data_loader import MalwareDataset
from model import MalwareDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_under_attack(model, test_loader, attack_fn, attack_kwargs, device):
    """
    Evaluates the model's accuracy on adversarially perturbed data.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.
        attack_fn: The CleverHans attack function (e.g., fast_gradient_method).
        attack_kwargs (dict): Dictionary of keyword arguments for the attack function.
        device (torch.device): The device to run evaluation on (CPU or GPU).

    Returns:
        float: The accuracy of the model on the adversarial examples.
    """
    model.eval() # Start in eval mode
    correct_predictions = 0
    total_samples = 0

    for (api_batch, global_batch), labels_batch in test_loader:
        api_batch = api_batch.to(device)
        global_batch = global_batch.to(device)
        labels_batch = labels_batch.to(device)

        # --- Generate Adversarial Examples ---
        # We need a wrapper for our model because CleverHans expects a model that
        # takes a single input and returns a 2D tensor of shape (batch, num_classes).
        def model_fn(x):
            # Get the original output, shape (batch_size)
            probs_malware = model(x, global_batch)
            
            # Create the probability for the benign class (class 0)
            probs_benign = 1.0 - probs_malware
            
            # Stack them into the shape CleverHans expects: (batch_size, 2)
            return torch.stack([probs_benign, probs_malware], dim=-1)

        # Add the ground-truth labels to the attack arguments. This is required.
        batch_attack_kwargs = attack_kwargs.copy()
        batch_attack_kwargs['y'] = labels_batch.long()

        # >>> THE FIX IS HERE <<<
        # Temporarily switch to train() mode for gradient calculation
        model.train()
        api_batch_adv = attack_fn(model_fn, api_batch, **batch_attack_kwargs)
        # Switch back to eval() mode for the actual prediction
        model.eval()
        # >>> END OF FIX <<<

        # --- Get Model Prediction ---
        # Ensure prediction is done in eval mode
        with torch.no_grad():
            outputs = model(api_batch_adv, global_batch)
            
        predicted = (outputs > 0.5).float()
        
        correct_predictions += (predicted == labels_batch).sum().item()
        total_samples += labels_batch.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy

def main(args):
    """Main function to orchestrate the attack and evaluation process."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}.")
        return
    model = MalwareDetectionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info("Model loaded successfully.")

    # --- Load Test Data ---
    splits_path = os.path.join(args.data_dir, 'dataset_splits.json')
    if not os.path.exists(splits_path):
        logging.error(f"Dataset splits file not found at {splits_path}.")
        return
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    test_dataset = MalwareDataset(splits['test']['files'], splits['test']['labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    logging.info(f"Test set loaded with {len(test_dataset)} samples.")

    # --- Run Attacks ---
    epsilons_to_test = [0.01, 0.03, 0.06,0.08, 0.1]
    results = {}
    
    print("\n--- Starting Adversarial Attacks ---")

    # FGSM Attack
    for eps in epsilons_to_test:
        logging.info(f"Running FGSM attack with epsilon = {eps}...")
        attack_kwargs = {'eps': eps, 'norm': np.inf}
        accuracy = evaluate_under_attack(model, test_loader, fast_gradient_method, attack_kwargs, device)
        results[f'FGSM (eps={eps})'] = accuracy
        logging.info(f"FGSM (eps={eps}) Accuracy: {accuracy:.4f}")

    # PGD Attack
    for eps in epsilons_to_test:
        logging.info(f"Running PGD attack with epsilon = {eps}...")
        attack_kwargs = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': 40, 'norm': np.inf}
        accuracy = evaluate_under_attack(model, test_loader, projected_gradient_descent, attack_kwargs, device)
        results[f'PGD (eps={eps})'] = accuracy
        logging.info(f"PGD (eps={eps}) Accuracy: {accuracy:.4f}")

    # --- Print Final Summary ---
    print("\n--- Adversarial Attack Benchmark Report ---")
    print(f"Model: {args.model_path}")
    print("-------------------------------------------")
    for attack_name, acc in results.items():
        print(f"{attack_name:<20} | Accuracy: {acc:.4f}")
    print("-------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a model against adversarial attacks.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'best_model.pth'),
                        help="Path to the trained 'best_model.pth' file.")
    parser.add_argument('--data_dir', type=str, default=script_dir,
                        help="Directory containing 'dataset_splits.json'.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for evaluation (smaller is often better for attacks).")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Disable CUDA even if it's available.")

    args = parser.parse_args()
    main(args)
