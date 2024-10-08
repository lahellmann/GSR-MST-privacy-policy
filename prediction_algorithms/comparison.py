import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the NumPy arrays
Keras_X_test = np.load('Keras_X_test.npy')
Keras_y_test = np.load('Keras_y_test.npy')
PyTorch_X_test = np.load('PyTorch_X_test.npy')
PyTorch_y_test = np.load('PyTorch_y_test.npy')
PyTorch_X_train = np.load('PyTorch_X_train.npy')

# If using TensorFlow/Keras, you can use them directly
# For PyTorch, convert them to tensors
X_test_torch = torch.Tensor(PyTorch_X_test)
y_test_torch = torch.Tensor(PyTorch_y_test)

def plotAll (ytest, keras_predictions, pytorch_predictions): # Plot actual vs predicted for both models on the same plot
    plt.figure(figsize=(12, 6))

    # Plot the actual values
    plt.plot(y_test, label='Actual', alpha=0.6)

    # Plot the Keras model predictions
    plt.plot(keras_predictions, label='Keras Predicted', alpha=0.6)

    # Plot the PyTorch model predictions
    plt.plot(pytorch_predictions, label='PyTorch Predicted', alpha=0.6)

    # Set the plot title and labels
    plt.title('Actual vs Predicted Stress Levels (Keras and PyTorch)')
    plt.xlabel('Sample Index')
    plt.ylabel('Stress Level')

    # Add a legend to differentiate between the models
    plt.legend()

    # Show the plot
    plt.show()
# LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the output from the last time step
        return out
    
# Load the saved Keras model
keras_model = tf.keras.models.load_model('lstm_model.h5')


# Evaluate the Keras model
keras_test_loss = keras_model.evaluate(Keras_X_test, Keras_y_test, verbose=0)
print(f'Keras Test Loss: {keras_test_loss:.4f}')

# Get predictions from the Keras model
keras_predictions = keras_model.predict(Keras_X_test)

# Instantiate the PyTorch model (same architecture as before)
input_size = PyTorch_X_train.shape[2]  # Number of features
hidden_size = 50
output_size = 1
num_layers = 1

pytorch_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
# Load the saved state dictionary into the PyTorch model
pytorch_model.load_state_dict(torch.load('lstm_model.pth'))

# Create PyTorch Dataset
test_dataset = TensorDataset(X_test_torch, y_test_torch)
# DataLoader for batching
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Loss and optimizer
criterion = nn.MSELoss()


# Ensure the model is in evaluation mode
pytorch_model.eval()

# Make predictions using the loaded PyTorch model
with torch.no_grad():
    predictions = []
    for X_batch, _ in test_loader:
        outputs = pytorch_model(X_batch)
        predictions.append(outputs)

# Concatenate predictions into a single tensor
pytorch_predictions = torch.cat(predictions).numpy()

# Evaluate the PyTorch model
pytorch_model.eval()  # Set the model to evaluation mode
test_loss = 0
predictions = []

with torch.no_grad():  # Disable gradient calculation for evaluation
    for X_batch, y_batch in test_loader:
        outputs = pytorch_model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        test_loss += loss.item()
        predictions.append(outputs)

test_loss /= len(test_loader)  # Average loss over the test set
print(f'PyTorch Test Loss: {test_loss:.4f}')

# Concatenate predictions into a single tensor
pytorch_predictions = torch.cat(predictions).numpy()

# Convert predictions and actual data to numpy arrays (if not already)
keras_y_test_np = np.array(Keras_y_test)  # Actual values
pytorch_y_test_np = np.array(PyTorch_y_test)  # Actual values
keras_predictions_np = np.array(keras_predictions).flatten()  # Keras predictions
pytorch_predictions_np = np.array(pytorch_predictions).flatten()  # PyTorch predictions

# Ensure all arrays have the same length
assert len(keras_y_test_np) == len(keras_predictions_np) == len(pytorch_predictions_np), "Mismatch in lengths of arrays"
# Ensure all arrays have the same length
assert len(pytorch_y_test_np) == len(keras_predictions_np) == len(pytorch_predictions_np), "Mismatch in lengths of arrays"

### 1. Count the number of data points that are exactly the same
# Exact match count for Keras
keras_exact_matches = np.sum(keras_y_test_np == keras_predictions_np)

# Exact match count for PyTorch
pytorch_exact_matches = np.sum(pytorch_y_test_np == pytorch_predictions_np)

print(f"Number of exact matches for Keras: {keras_exact_matches}")
print(f"Number of exact matches for PyTorch: {pytorch_exact_matches}")

### 2. Calculate variance (Mean Squared Error can be used to measure variance)
# Variance for Keras (Mean Squared Difference)
keras_variance = np.mean((keras_y_test_np - keras_predictions_np) ** 2)

# Variance for PyTorch (Mean Squared Difference)
pytorch_variance = np.mean((pytorch_y_test_np - pytorch_predictions_np) ** 2)

print(f"Variance (MSE) for Keras: {keras_variance}")
print(f"Variance (MSE) for PyTorch: {pytorch_variance}")

### 3. Calculate the maximum absolute difference
# Max difference for Keras
keras_max_diff = np.max(np.abs(keras_y_test_np - keras_predictions_np))

# Max difference for PyTorch
pytorch_max_diff = np.max(np.abs(pytorch_y_test_np - pytorch_predictions_np))

print(f"Maximum difference for Keras: {keras_max_diff}")
print(f"Maximum difference for PyTorch: {pytorch_max_diff}")


# Calculate Mean Absolute Percentage Error for Keras
keras_mape = np.mean(np.abs((keras_y_test_np - keras_predictions_np) / keras_y_test_np)) * 100

# Calculate Mean Absolute Percentage Error for PyTorch
pytorch_mape = np.mean(np.abs((pytorch_y_test_np - pytorch_predictions_np) / pytorch_y_test_np)) * 100

print(f"Keras Model MAPE: {keras_mape:.2f}%")
print(f"PyTorch Model MAPE: {pytorch_mape:.2f}%")

from sklearn.metrics import r2_score

# Calculate R-squared for Keras
keras_r2 = r2_score(keras_y_test_np, keras_predictions_np)

# Calculate R-squared for PyTorch
pytorch_r2 = r2_score(pytorch_y_test_np, pytorch_predictions_np)

# Convert R-squared to percentage
keras_r2_percent = keras_r2 * 100
pytorch_r2_percent = pytorch_r2 * 100

print(f"Keras Model R-squared: {keras_r2_percent:.2f}%")
print(f"PyTorch Model R-squared: {pytorch_r2_percent:.2f}%")
