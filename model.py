

"""
model.py

This script defines the hybrid deep learning architecture for malware detection,
inspired by the research paper (1907.07352v5.pdf) and adapted for the specific
feature structure from our `extractor_nn.py` script.

The model consists of two main branches:
1.  A Sequential Branch to process the sequence of API calls (`_apis.npy`). This
    branch uses Gated Convolutional Neural Networks (Gated CNNs) to learn local
    patterns, followed by a Bidirectional LSTM (Bi-LSTM) to capture long-range
    sequential dependencies.
2.  A Global Branch to process the summary feature vector (`_globals.npy`). This
    is a simple Multi-Layer Perceptron (MLP).

The outputs of both branches are concatenated and passed through a final
classifier head to produce a single prediction (malware or benign).
"""
import torch
import torch.nn as nn

class GatedCNN(nn.Module):
    """
    Implements the Gated CNN block described in the paper.
    This involves two parallel 1D convolutions. The output of one is passed
    through a sigmoid function to act as a 'gate' for the other.
    This version includes manual padding to ensure the output sequence length
    is the same as the input sequence length, regardless of kernel size.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(GatedCNN, self).__init__()
        # Set padding to 0 in the conv layer as we'll handle it manually
        self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.sigmoid = nn.Sigmoid()
        # Store kernel size to calculate padding in the forward pass
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Performs a forward pass with manual 'same' padding.
        """
        # x shape: (batch_size, in_channels, sequence_length)
        
        # Calculate the total padding required to keep the sequence length the same
        padding_total = self.kernel_size - 1
        # Split padding for left and right sides (handles even kernels correctly)
        padding_left = padding_total // 2
        padding_right = padding_total - padding_left
        
        # Apply manual padding to the sequence dimension (last dimension)
        x_padded = torch.nn.functional.pad(x, (padding_left, padding_right))
        
        out = self.conv_out(x_padded)
        gate = self.sigmoid(self.conv_gate(x_padded))
        
        return out * gate # Element-wise multiplication

class MalwareDetectionModel(nn.Module):
    """
    The main hybrid model for malware detection.
    """
    def __init__(self, api_feature_dim=118, global_feature_dim=8, lstm_hidden_dim=100, cnn_filters=128, classifier_hidden_dim=64, dropout_rate=0.5):
        """
        Initializes the layers of the two-branch model.

        Args:
            api_feature_dim (int): The number of features for each API call in the sequence (118).
            global_feature_dim (int): The number of global summary features (8).
            lstm_hidden_dim (int): The number of hidden units in the Bi-LSTM.
            cnn_filters (int): The number of output filters for the Gated CNNs.
            classifier_hidden_dim (int): The number of units in the hidden dense layer of the classifier.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(MalwareDetectionModel, self).__init__()

        # --- Sequential Branch ---
        self.seq_bn1 = nn.BatchNorm1d(api_feature_dim)
        
        # Two Gated CNNs with different kernel sizes as per the paper's ablation study
        self.gated_cnn2 = GatedCNN(api_feature_dim, cnn_filters, kernel_size=2)
        self.gated_cnn3 = GatedCNN(api_feature_dim, cnn_filters, kernel_size=3)
        
        # The input to the next layer is the concatenated output of the two Gated CNNs
        self.seq_bn2 = nn.BatchNorm1d(cnn_filters * 2)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        # Global Max Pooling will be applied in the forward pass

        # --- Global Features Branch ---
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        # --- Classifier Head ---
        # The input to the classifier is the concatenated output of:
        # 1. Bi-LSTM (after pooling): lstm_hidden_dim * 2
        # 2. Global MLP: 16
        classifier_input_dim = (lstm_hidden_dim * 2) + 16
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, 1) # Single output for binary classification
        )
        
        self.final_activation = nn.Sigmoid()

    def forward(self, api_sequence, global_features):
        """
        Defines the forward pass of the model.

        Args:
            api_sequence (torch.Tensor): Tensor of shape (batch_size, seq_len, api_feature_dim).
            global_features (torch.Tensor): Tensor of shape (batch_size, global_feature_dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) with the model's prediction.
        """
        # --- Process Sequential Branch ---
        # Reshape for Conv1D: (batch_size, features, seq_len)
        api_sequence = api_sequence.permute(0, 2, 1)
        seq_out = self.seq_bn1(api_sequence)
        
        # Apply Gated CNNs
        cnn2_out = self.gated_cnn2(seq_out)
        cnn3_out = self.gated_cnn3(seq_out)
        
        # Concatenate along the feature dimension
        seq_out = torch.cat((cnn2_out, cnn3_out), dim=1)
        seq_out = self.seq_bn2(seq_out)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        seq_out = seq_out.permute(0, 2, 1)
        
        # Apply Bi-LSTM
        lstm_out, _ = self.bilstm(seq_out)
        
        # Apply Global Max Pooling over the sequence dimension
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim * 2)
        # We pool over dimension 1 (the sequence_length)
        seq_vector = torch.max(lstm_out, dim=1)[0]

        # --- Process Global Branch ---
        global_vector = self.global_mlp(global_features)

        # --- Combine and Classify ---
        combined_vector = torch.cat((seq_vector, global_vector), dim=1)
        
        logits = self.classifier(combined_vector)
        prediction = self.final_activation(logits)
        
        return prediction.squeeze(1) # Squeeze to (batch_size) for BCELoss

if __name__ == '__main__':
    # Example of how to instantiate and use the model
    print("--- Model Instantiation Example ---")
    
    # Create dummy input tensors
    batch_size = 4
    seq_length = 500
    api_dim = 118
    global_dim = 8
    
    dummy_api_seq = torch.randn(batch_size, seq_length, api_dim)
    dummy_global_feat = torch.randn(batch_size, global_dim)
    
    # Instantiate the model
    model = MalwareDetectionModel()
    print(model)
    
    # Perform a forward pass
    output = model(dummy_api_seq, dummy_global_feat)
    
    print(f"\nInput API sequence shape: {dummy_api_seq.shape}")
    print(f"Input global features shape: {dummy_global_feat.shape}")
    print(f"Output prediction shape: {output.shape}")
    print(f"Example output: {output.detach().numpy()}")
    print("\nExample finished.")

