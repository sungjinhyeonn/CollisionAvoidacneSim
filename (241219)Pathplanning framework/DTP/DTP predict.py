import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PyTorch model (GRU-based S-TGP)
class STGPModel(nn.Module):
    def __init__(self, input_dim):
        super(STGPModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=4, padding=4)
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)  # Output: next x, y

    def forward(self, x):
        x = x.transpose(1, 2)  # Conv1D expects (batch_size, input_dim, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(1, 2)  # Transpose back to (batch_size, sequence_length, channels)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.fc(x[:, -1, :])  # Use the last timestep output
        return x

# Load the trained model
def load_model(model_path, input_dim):
    model = STGPModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

# Prepare the data using a sequence length of 10
def prepare_data_with_sequence_length(agent_data, sequence_length=10):
    input_data = agent_data[['x', 'y', 'yaw', 'linear_velocity', 'angular_velocity']].values
    X_seq = []
    
    for i in range(len(input_data) - sequence_length):
        X_seq.append(input_data[i:i + sequence_length])
    
    return np.array(X_seq)

# Load new data from a CSV file and preprocess it into sequences of 10 timesteps
def load_new_data_with_sequences(file_path, sequence_length=10):
    new_data = pd.read_csv(file_path)
    input_sequences = prepare_data_with_sequence_length(new_data, sequence_length)
    return torch.tensor(input_sequences, dtype=torch.float32), new_data[['x', 'y']].values

# Visualization function
def visualize_prediction(original_data, predicted_data):
    plt.figure(figsize=(8, 6))
    plt.plot(original_data[:, 0], original_data[:, 1], 'bo-', label='Original Path (x, y)')
    plt.plot(predicted_data[:, 0], predicted_data[:, 1], 'ro-', label='Predicted Path (x, y)')

    plt.title('Path Prediction vs Original')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main prediction function using the loaded model and new data
def predict_entire_sequence_with_sequence_length(model_path, new_data_path, sequence_length=10):
    # Load the trained model
    model = load_model(model_path, input_dim=5)

    # Load and preprocess new data into sequences of 10 timesteps
    new_X, original_xy = load_new_data_with_sequences(new_data_path, sequence_length)

    # Initialize list to store predicted values
    predicted_xy_list = []

    # Predict the next step based on each sequence of 10 timesteps
    with torch.no_grad():
        for i in range(len(new_X)):
            input_seq = new_X[i].unsqueeze(0)  # Add batch dimension
            predicted_xy = model(input_seq)
            predicted_xy_list.append(predicted_xy.squeeze().cpu().numpy())

    # Convert list to numpy array for visualization
    predicted_xy_array = np.array(predicted_xy_list)

    # Visualize the result
    visualize_prediction(original_xy[sequence_length:], predicted_xy_array)

# Example usage
if __name__ == "__main__":
    # Path to the trained model and new data
    model_path = 'trained_stgp_model.pth'  # Update this path with your model
    new_data_path = 'Agent_31_Maneuver\Agent_35_Maneuver.csv'  # Update this path with your new data file

    # Predict next (x, y) using the trained model and visualize the result with sequence length of 10
    predict_entire_sequence_with_sequence_length(model_path, new_data_path, sequence_length=10)
