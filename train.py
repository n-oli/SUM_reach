
"""
train.py

This is the main script for training the malware detection model.

It performs the following steps:
1.  Parses command-line arguments for hyperparameters like learning rate, epochs, etc.
2.  Loads the training and validation dataset splits created by `data_loader.py`.
3.  Initializes the `MalwareDataset` and `DataLoader` for both training and validation sets.
4.  Initializes the `MalwareDetectionModel`, the Adam optimizer, and the binary
    cross-entropy loss function.
5.  Runs the main training loop for the specified number of epochs.
6.  Within the loop, it trains the model on the training set and evaluates its
    performance on the validation set after each epoch.
7.  Implements a "save-on-best" mechanism, where the model's state is saved to
    `best_model.pth` only if its validation accuracy improves.
"""
import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from data_loader import MalwareDataset
from model import MalwareDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, optimizer, criterion, device):
    """
    Runs a single epoch of training.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run training on (CPU or GPU).

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for (api_batch, global_batch), labels_batch in train_loader:
        # Move data to the specified device
        api_batch = api_batch.to(device)
        global_batch = global_batch.to(device)
        labels_batch = labels_batch.to(device)

        # --- Forward pass ---
        optimizer.zero_grad()
        outputs = model(api_batch, global_batch)
        loss = criterion(outputs, labels_batch)

        # --- Backward pass and optimization ---
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        total_loss += loss.item() * api_batch.size(0)
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == labels_batch).sum().item()
        total_samples += labels_batch.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to be evaluated.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing the average loss and accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
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

def main(args):
    """Main function to orchestrate the training process."""
    
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    splits_path = os.path.join(args.data_dir, 'dataset_splits.json')
    if not os.path.exists(splits_path):
        logging.error(f"Dataset splits file not found at {splits_path}.")
        logging.error("Please run data_loader.py first to generate the splits.")
        return
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    train_dataset = MalwareDataset(splits['train']['files'], splits['train']['labels'])
    val_dataset = MalwareDataset(splits['validation']['files'], splits['validation']['labels'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Data loaders created successfully.")

    # --- Initialize Model, Optimizer, and Loss ---
    model = MalwareDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    logging.info("Model, optimizer, and loss function initialized.")
    
    # --- Training Loop ---
    best_val_accuracy = 0.0
    output_model_path = os.path.join(args.output_dir, 'best_model.pth')
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save the model if validation accuracy has improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), output_model_path)
            logging.info(f"Validation accuracy improved. Saving model to {output_model_path}")

    logging.info("Training finished.")
    logging.info(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Malware Detection Model")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_dir', type=str, default=script_dir,
                        help="Directory containing 'dataset_splits.json'. Defaults to the script's directory.")
    parser.add_argument('--output_dir', type=str, default=script_dir,
                        help="Directory to save the best model. Defaults to the script's directory.")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of worker processes for the DataLoader.")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Disable CUDA even if it's available.")

    args = parser.parse_args()
    main(args)
