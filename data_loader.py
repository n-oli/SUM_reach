

"""
data_loader.py

This script handles the creation of dataset splits and provides a memory-efficient
PyTorch Dataset and DataLoader for training the malware detection model.

It performs a one-time scan of the feature directory to create stratified train,
validation, and test sets, saving the file paths to JSON files. This prevents
data leakage and ensures the test set remains unseen until final evaluation.

The MalwareDataset class loads .npy feature files on-the-fly, ensuring that
the entire dataset is never loaded into memory at once.
"""
import os
import json
import logging
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Define which directories represent malware and which are benign.
MALWARE_CLASSES = ['report_backdoor', 'report_ransomware']
BENIGN_CLASSES = ['report_clean']
LABEL_MAP = {"benign": 0, "malware": 1}

def create_splits(feature_dir, output_dir, test_size=0.2, val_size=0.2):
    """
    Scans the feature directory, creates stratified train/validation/test splits
    based on the specified malware and benign classes, and saves the file paths
    and labels into JSON files.

    This function should only be run once to ensure the test set remains consistent.

    Args:
        feature_dir (str): The path to the directory containing the feature subdirectories
                           (e.g., 'features_nn').
        output_dir (str): The directory where the split files will be saved (e.g., 'Neural_nets').
        test_size (float): The proportion of the dataset to allocate to the test set.
        val_size (float): The proportion of the remaining (non-test) data to allocate
                          to the validation set.
    """
    logging.info(f"Scanning feature directory: {feature_dir}")
    all_files = []
    all_labels = []

    # Process benign files
    for class_name in BENIGN_CLASSES:
        class_dir = os.path.join(feature_dir, class_name)
        api_files = glob(os.path.join(class_dir, '*_apis.npy'))
        for api_file in api_files:
            base_name = api_file.replace('_apis.npy', '')
            global_file = base_name + '_globals.npy'
            if os.path.exists(global_file):
                all_files.append({'apis': api_file, 'globals': global_file})
                all_labels.append(LABEL_MAP['benign'])

    # Process malware files
    for class_name in MALWARE_CLASSES:
        class_dir = os.path.join(feature_dir, class_name)
        api_files = glob(os.path.join(class_dir, '*_apis.npy'))
        for api_file in api_files:
            base_name = api_file.replace('_apis.npy', '')
            global_file = base_name + '_globals.npy'
            if os.path.exists(global_file):
                all_files.append({'apis': api_file, 'globals': global_file})
                all_labels.append(LABEL_MAP['malware'])

    if not all_files:
        logging.error("No feature files found. Please run the extractor script first.")
        return

    logging.info(f"Found {len(all_files)} total samples.")
    logging.info(f"Benign samples: {all_labels.count(0)}, Malware samples: {all_labels.count(1)}")

    # Create initial split for the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_files, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )

    # Create the second split for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )

    logging.info(f"Train set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")
    logging.info(f"Test set size: {len(X_test)}")

    # Prepare data for JSON serialization
    splits = {
        'train': {'files': X_train, 'labels': y_train},
        'validation': {'files': X_val, 'labels': y_val},
        'test': {'files': X_test, 'labels': y_test}
    }

    os.makedirs(output_dir, exist_ok=True)
    splits_path = os.path.join(output_dir, 'dataset_splits.json')
    test_set_path = os.path.join(output_dir, 'test_set_paths.json')
    label_map_path = os.path.join(output_dir, 'label_map.json')

    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
    with open(test_set_path, 'w') as f:
        json.dump({'files': X_test, 'labels': y_test}, f, indent=4)
    with open(label_map_path, 'w') as f:
        json.dump(LABEL_MAP, f, indent=4)

    logging.info(f"Dataset splits saved successfully to {output_dir}")


class MalwareDataset(Dataset):
    """
    A memory-efficient PyTorch Dataset for loading malware features.

    This dataset class takes a list of file paths and their corresponding labels.
    It loads the actual .npy data for a sample only when that sample is requested
    (in the `__getitem__` method), avoiding high memory usage. It also handles
    padding/truncating of API sequences to ensure consistent tensor sizes for batching.
    """
    def __init__(self, file_paths, labels, api_seq_len=500):
        """
        Args:
            file_paths (list of dict): A list where each element is a dictionary
                                      like {'apis': path, 'globals': path}.
            labels (list): A list of integer labels corresponding to the file_paths.
            api_seq_len (int): The fixed length to which API sequences will be
                               padded or truncated.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.api_seq_len = api_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads, processes, and returns a single sample from the dataset.
        """
        paths = self.file_paths[idx]
        label = self.labels[idx]

        # Load the feature arrays from disk
        api_features = np.load(paths['apis'])
        global_features = np.load(paths['globals'])

        # Pad or truncate the API sequence
        seq_len, num_features = api_features.shape
        if seq_len > self.api_seq_len:
            # Truncate
            api_features = api_features[:self.api_seq_len, :]
        elif seq_len < self.api_seq_len:
            # Pad with zeros
            padding = np.zeros((self.api_seq_len - seq_len, num_features))
            api_features = np.vstack((api_features, padding))

        # Convert to PyTorch tensors
        api_tensor = torch.from_numpy(api_features).float()
        global_tensor = torch.from_numpy(global_features).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return (api_tensor, global_tensor), label_tensor


if __name__ == '__main__':
    # This block will be executed when the script is run directly.
    # It generates the dataset split files needed for training and evaluation.
    # NOTE: This should only be run ONCE.
    
    # Assumes the script is in 'Neural_nets' and the features are in the parent dir.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_directory = os.path.join(current_dir, '..', 'features_nn')
    output_directory = current_dir

    if not os.path.exists(os.path.join(output_directory, 'dataset_splits.json')):
        create_splits(feature_dir=feature_directory, output_dir=output_directory)
    else:
        logging.warning("Split files already exist. Skipping creation.")
        logging.warning("To regenerate splits, delete 'dataset_splits.json' and 'test_set_paths.json'.")

    # Example of how to use the MalwareDataset and DataLoader
    logging.info("\n--- DataLoader Example ---")
    with open(os.path.join(output_directory, 'dataset_splits.json'), 'r') as f:
        splits_data = json.load(f)
    
    train_files = splits_data['train']['files']
    train_labels = splits_data['train']['labels']

    train_dataset = MalwareDataset(train_files, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Fetch one batch to demonstrate
    (api_batch, global_batch), labels_batch = next(iter(train_loader))
    logging.info(f"API sequence batch shape: {api_batch.shape}")
    logging.info(f"Global features batch shape: {global_batch.shape}")
    logging.info(f"Labels batch shape: {labels_batch.shape}")
    logging.info("Example finished.")

