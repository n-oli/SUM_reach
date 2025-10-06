

"""
evaluate.py

This script evaluates the final performance of a trained malware detection model
on the unseen test set.

It performs the following steps:
1.  Loads the best-performing model saved by `train.py` (e.g., 'best_model.pth').
2.  Loads the test set file paths and labels from 'dataset_splits.json'.
3.  Initializes a DataLoader for the test set.
4.  Performs inference on the entire test set.
5.  Calculates and prints a comprehensive classification report, including:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
6.  Generates and saves a confusion matrix visualization to 'confusion_matrix.png'.

This script requires scikit-learn, seaborn, and matplotlib for metrics and plotting.
Install them with:
pip install scikit-learn seaborn matplotlib
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import MalwareDataset
from model import MalwareDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test(model, test_loader, device):
    """
    Evaluates the model on the test set and returns performance metrics.

    Args:
        model (nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.
        device (torch.device): The device to run evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing (all true labels, all predicted labels).
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (api_batch, global_batch), labels_batch in test_loader:
            api_batch = api_batch.to(device)
            global_batch = global_batch.to(device)
            
            outputs = model(api_batch, global_batch)
            predicted = (outputs > 0.5).float().cpu().numpy()
            
            all_labels.extend(labels_batch.numpy())
            all_predictions.extend(predicted)

    return np.array(all_labels), np.array(all_predictions)

def plot_confusion_matrix(cm, class_names, output_filename):
    """
    Renders and saves the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_filename)
    logging.info(f"Confusion matrix saved to {output_filename}")

def main(args):
    """Main function to orchestrate the evaluation process."""
    
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}. Please run train.py first.")
        return
    model = MalwareDetectionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info("Model loaded successfully.")

    # --- Load Test Data ---
    splits_path = os.path.join(args.data_dir, 'dataset_splits.json')
    if not os.path.exists(splits_path):
        logging.error(f"Dataset splits file not found at {splits_path}. Please run data_loader.py first.")
        return
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    test_dataset = MalwareDataset(splits['test']['files'], splits['test']['labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info(f"Test set loaded with {len(test_dataset)} samples.")

    # --- Perform Evaluation ---
    true_labels, pred_labels = test(model, test_loader, device)

    # --- Calculate and Display Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    
    print("\n--- Test Set Evaluation Report ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("----------------------------------\n")

    # --- Generate Confusion Matrix Plot ---
    label_map_path = os.path.join(args.data_dir, 'label_map.json')
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    # Sort by value to ensure 'benign' is first (0), 'malware' is second (1)
    class_names = sorted(label_map.keys(), key=label_map.get)
    
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Malware Detection Model on the test set.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_dir', type=str, default=script_dir,
                        help="Directory containing 'dataset_splits.json'. Defaults to the script's directory.")
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'best_model.pth'),
                        help="Path to the trained 'best_model.pth' file.")
    parser.add_argument('--output_dir', type=str, default=script_dir,
                        help="Directory to save the confusion matrix plot. Defaults to the script's directory.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of worker processes for the DataLoader.")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Disable CUDA even if it's available.")

    args = parser.parse_args()
    main(args)
