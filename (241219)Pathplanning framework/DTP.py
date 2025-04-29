import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Function to load agent data files
def load_agent_data(file_directory):
    agent_data_list = []
    for file_name in os.listdir(file_directory):
        if file_name.startswith('Obstacle_') and file_name.endswith('.csv'):
            file_path = os.path.join(file_directory, file_name)
            agent_data = pd.read_csv(file_path)
            agent_data_list.append(agent_data)
    if not agent_data_list:
        raise ValueError("No agent files found in the directory")
    return agent_data_list

# Prepare data for training: create X (inputs) and y (targets) from agent data
def prepare_data_for_training(agent_data_list, sequence_length=10):
    X = []
    y = []
    
    for agent_data in agent_data_list:
        # Input: current x, y, yaw, linear_velocity, angular_velocity
        input_data = agent_data[['x', 'y', 'yaw', 'linear_velocity', 'angular_velocity']].values
        
        for i in range(len(input_data) - sequence_length):
            # Create input sequences of 10 timesteps
            X.append(input_data[i:i + sequence_length])
            # The target is the next timestep's x, y
            y.append(agent_data[['x', 'y']].iloc[i + sequence_length].values)
    
    return np.array(X), np.array(y)

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
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(2, 1)  # Transpose to (batch_size, input_dim, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(2, 1)  # Transpose back to (batch_size, sequence_length, channels)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.fc(x[:, -1, :])  # Use the last timestep output
        return x

# Train the model using the given data
def train_model(X_train, y_train, input_dim, save_path):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = STGPModel(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    batch_size = 32

    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # Sequence data is already in the correct shape
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

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

# Main function to load data, train, predict, and visualize
def main(folder_path, model_save_path):
    # Load all data from the folder
    all_agent_data = load_agent_data(folder_path)

    # Prepare data with 10 timesteps input sequences
    X_train, y_train = prepare_data_for_training(all_agent_data, sequence_length=10)

    # Train the model
    model = train_model(X_train, y_train, input_dim=5, save_path=model_save_path)

    # Make predictions with the trained model
    X_test = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        predicted = model(X_test).numpy()

    # Visualize the predictions vs original
    visualize_prediction(y_train, predicted)

# Example usage
if __name__ == "__main__":
    # Define the folder containing agent data and model save path
    folder_path = 'log\dwa'  # Update this with your folder path
    model_save_path = 'trained_stgp_model.pth'  # Path to save the trained model

    # Run the main function to train the model and visualize results
    main(folder_path, model_save_path)
