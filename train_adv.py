
"""
train_adv.py

This script performs adversarial training to create a robust malware detection model.

It enhances the standard training process by incorporating adversarial examples
generated on-the-fly for each training batch.

Key Features:
-   Accepts command-line arguments to specify the attack type (`--attack`),
    the perturbation magnitude (`--epsilon`), and an output directory.
-   Uses the CleverHans library to generate adversarial examples (FGSM or PGD).
-   The training loop calculates loss on both the original clean batch and the
    newly generated adversarial batch, then updates the model based on the combined loss.
-   Continues to use a clean validation set to save the best-performing model,
    ensuring that robustness doesn't come at a complete loss of normal performance.
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- CleverHans Imports ---
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# Import custom modules
from data_loader import MalwareDataset
from model import MalwareDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for (api_batch, global_batch), labels_batch in val_loader:
            api_batch = api_batch.to(device)
            global_batch = global_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(api_batch, global_batch)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item() * api_batch.size(0)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels_batch).sum().item()
            total_samples += labels_batch.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def train_adversarial(model, train_loader, optimizer, criterion, device, attack_fn, attack_kwargs):
    """
    Runs a single epoch of adversarial training.

    For each batch, it generates adversarial examples and trains the model on both
    the clean and adversarial data.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for (api_batch, global_batch), labels_batch in train_loader:
        api_batch = api_batch.to(device)
        global_batch = global_batch.to(device)
        labels_batch = labels_batch.to(device)

        # --- 1. Generate Adversarial Examples ---
        # Create the model wrapper required by CleverHans
        def model_fn(x):
            probs_malware = model(x, global_batch)
            probs_benign = 1.0 - probs_malware
            return torch.stack([probs_benign, probs_malware], dim=-1)

        # Set attack kwargs for the current batch
        batch_attack_kwargs = attack_kwargs.copy()
        batch_attack_kwargs['y'] = labels_batch.long()
        
        # Generate the adversarial version of the API sequence batch
        api_batch_adv = attack_fn(model_fn, api_batch, **batch_attack_kwargs)

        # --- 2. Forward pass on both clean and adversarial data ---
        optimizer.zero_grad()
        
        # Clean data
        outputs_clean = model(api_batch, global_batch)
        loss_clean = criterion(outputs_clean, labels_batch)
        
        # Adversarial data
        outputs_adv = model(api_batch_adv, global_batch)
        loss_adv = criterion(outputs_adv, labels_batch)

        # Combine the losses
        loss = loss_clean + loss_adv

        # --- 3. Backward pass and optimization ---
        loss.backward()
        optimizer.step()

        # --- 4. Statistics (based on clean data performance) ---
        total_loss += loss_clean.item() * api_batch.size(0)
        predicted = (outputs_clean > 0.5).float()
        correct_predictions += (predicted == labels_batch).sum().item()
        total_samples += labels_batch.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def main(args):
    """Main function to orchestrate the adversarial training process."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"--- Starting Adversarial Training ---")
    logging.info(f"Device: {device} | Attack: {args.attack.upper()} | Epsilon: {args.epsilon}")
    logging.info(f"Output will be saved to: {args.output_dir}")

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Data ---
    splits_path = os.path.join(args.data_dir, 'dataset_splits.json')
    if not os.path.exists(splits_path):
        logging.error(f"Dataset splits file not found at {splits_path}. Run data_loader.py first.")
        return
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Use drop_last=True to prevent batch size of 1, which causes issues with BatchNorm in train mode
    train_dataset = MalwareDataset(splits['train']['files'], splits['train']['labels'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    val_dataset = MalwareDataset(splits['validation']['files'], splits['validation']['labels'])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Data loaders created successfully.")

    # --- Initialize Model, Optimizer, and Loss ---
    model = MalwareDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    logging.info("Model, optimizer, and loss function initialized.")

    # --- Configure Attack ---
    if args.attack == 'fgsm':
        attack_fn = fast_gradient_method
        attack_kwargs = {'eps': args.epsilon, 'norm': np.inf}
    elif args.attack == 'pgd':
        attack_fn = projected_gradient_descent
        attack_kwargs = {'eps': args.epsilon, 'eps_iter': 0.01, 'nb_iter': 40, 'norm': np.inf}
    else:
        logging.error(f"Unknown attack type: {args.attack}")
        return
    
    # --- Training Loop ---
    best_val_accuracy = 0.0
    output_model_path = os.path.join(args.output_dir, 'best_model.pth')

    logging.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_adversarial(model, train_loader, optimizer, criterion, device, attack_fn, attack_kwargs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), output_model_path)
            logging.info(f"Validation accuracy improved. Saving model to {output_model_path}")

    logging.info("Adversarial training finished.")
    logging.info(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform adversarial training on the Malware Detection Model.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_dir', type=str, default=script_dir,
                        help="Directory containing 'dataset_splits.json'.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the best adversarially trained model.")
    parser.add_argument('--attack', type=str, required=True, choices=['fgsm', 'pgd'],
                        help="The adversarial attack to train against.")
    parser.add_argument('--epsilon', type=float, required=True,
                        help="The perturbation magnitude (epsilon) for the attack.")
    
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of worker processes for DataLoader.")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA.")

    args = parser.parse_args()
    main(args)
