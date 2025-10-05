"""
predict.py

This script provides a command-line interface for classifying a single sample
using its pre-extracted feature files (`_apis.npy` and `_globals.npy`).

It performs the following steps:
1.  Loads the best-performing model state from `best_model.pth`.
2.  Takes paths to the `_apis.npy` and `_globals.npy` files as arguments.
3.  Loads the feature arrays.
4.  Pads the API sequence to the required length.
5.  Feeds the features into the model for inference.
6.  Prints the classification result (Benign or Malware) and the model's
    confidence score.
"""
import os
import argparse
import logging
import torch
import numpy as np

# Import the model architecture
from model import MalwareDetectionModel

# Configuration constant must match the training configuration
API_SEQ_LEN = 500

def predict(model, device, api_path, global_path):
    """
    Loads pre-extracted features from .npy files and returns a prediction.
    """
    # 1. Load the feature arrays
    try:
        api_features = np.load(api_path)
        global_features = np.load(global_path)
    except Exception as e:
        logging.error(f"Failed to load .npy files: {e}")
        return

    # 2. Pad/truncate the API sequence
    seq_len, num_features = api_features.shape
    if seq_len > API_SEQ_LEN:
        api_features = api_features[:API_SEQ_LEN, :]
    elif seq_len < API_SEQ_LEN:
        padding = np.zeros((API_SEQ_LEN - seq_len, num_features))
        api_features = np.vstack((api_features, padding))

    # 3. Convert to tensors and add batch dimension
    api_tensor = torch.from_numpy(api_features).float().unsqueeze(0).to(device)
    global_tensor = torch.from_numpy(global_features).float().unsqueeze(0).to(device)

    # 4. Perform inference
    model.eval()
    with torch.no_grad():
        output = model(api_tensor, global_tensor)
        probability = output.item()

    return probability


def main(args):
    """Main function to load the model and run prediction."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}")
        logging.error("Please train the model first using train.py")
        return

    model = MalwareDetectionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info("Model loaded successfully.")

    # --- Run Prediction ---
    if not os.path.exists(args.api_file_path):
        logging.error(f"Input API feature file not found: {args.api_file_path}")
        return
    if not os.path.exists(args.global_file_path):
        logging.error(f"Input global feature file not found: {args.global_file_path}")
        return

    probability = predict(model, device, args.api_file_path, args.global_file_path)
    
    if probability is not None:
        label = "Malware" if probability > 0.5 else "Benign"
        print("\n--- Prediction Result ---")
        print(f"Sample:        {os.path.basename(args.api_file_path).replace('_apis.npy', '')}")
        print(f"Classification:  {label}")
        print(f"Confidence:      {probability:.4f}")
        print("-------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a sample using pre-extracted .npy feature files.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, 'best_model.pth')

    parser.add_argument('api_file_path', type=str,
                        help="Path to the '_apis.npy' feature file.")
    parser.add_argument('global_file_path', type=str,
                        help="Path to the '_globals.npy' feature file.")
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help=f"Path to the trained model file. Defaults to {default_model_path}")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Disable CUDA even if it's available.")

    args = parser.parse_args()
    main(args)