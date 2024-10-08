import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load all sheets from the Excel file
file_path = r'C:\Users\Laura\OneDrive\Desktop\UOS\Bachelor Thesis\Code_VS\hrv stress labels.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)

# Combine all sheets into one DataFrame
data_list = []
for sheet_name, sheet_data in sheets.items():
    sheet_data['participant_id'] = sheet_name  # Assign the sheet name as a participant ID
    data_list.append(sheet_data)

# Combine the data from all participants
data = pd.concat(data_list, ignore_index=True)

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
data[['HR', 'RMSSD', 'SCL']] = imputer.fit_transform(data[['HR', 'RMSSD', 'SCL']])

# Define weights based on literature
w_hr = 0.2
w_rmssd = 0.4
w_scl = 0.4

# Pre-compute max values for normalization
max_rmssd = data['RMSSD'].max()
max_scl = data['SCL'].max()

# Function to calculate stress score
def calculate_weighted_stress_score(row):
    hr_score = (row['HR'] / 100) if row['HR'] <= 100 else (100 / row['HR'])  # Scale HR
    return (w_hr * hr_score +
            w_rmssd * (row['RMSSD'] / max_rmssd) +  
            w_scl * (row['SCL'] / max_scl))  

# Apply the function to calculate stress score for each row
data['stress_score'] = data.apply(calculate_weighted_stress_score, axis=1)

# Normalize the data
features = data[['HR', 'RMSSD', 'SCL']]
target = data['stress_score']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Function to create sequences from data
def create_sequences(data, target, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(target[i + sequence_length])  # Predict next value
    return np.array(sequences), np.array(labels)

# Parameters
initial_sequence_length = 300  # 5 minutes (assuming 1 value per second)
prediction_length = 5  # Predict the next 5 seconds
desired_r_squared = 0.90  # Target R-squared value
current_r_squared = 0.0  # Initialize current R-squared
max_iterations = 10  # Maximum number of iterations to avoid infinite loop
iteration = 0  # Track the number of iterations

# Initialize a list to track R-squared values and sequence lengths
r_squared_values = []
sequence_lengths = []

# Function to calculate R-squared
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    residuals = y - predictions.flatten()
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Loop until the desired R-squared value is achieved or max iterations are reached
while current_r_squared < desired_r_squared and iteration < max_iterations:
    # Create sequences
    X, y = create_sequences(scaled_features, target.values, initial_sequence_length)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    current_r_squared = evaluate_model(model, X_test, y_test)
    print(f'Iteration {iteration + 1}, Sequence Length: {initial_sequence_length}, R-squared: {current_r_squared:.2f}')

    # Track the R-squared value and sequence length
    r_squared_values.append(current_r_squared)
    sequence_lengths.append(initial_sequence_length)

    # Dynamically adjust the sequence length if R-squared is not satisfactory
    initial_sequence_length += 300  # Incrementally increase input length by 5 minutes (300 seconds)
    iteration += 1  # Increment iteration

# Plot R-squared over time with the sequence lengths
plt.figure(figsize=(12, 6))
plt.plot(sequence_lengths, r_squared_values, marker='o')
plt.title('R-squared Improvement Over Time')
plt.xlabel('Sequence Length (seconds)')
plt.ylabel('R-squared Value')
plt.grid(True)
plt.show()

# Final Evaluation and Predictions
loss = model.evaluate(X_test, y_test)
print(f'Final Test Loss: {loss:.4f}')

# Make final predictions
final_predictions = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(final_predictions, label='Predicted')
plt.title('Actual vs Predicted Stress Levels')
plt.xlabel('Sample Index')
plt.ylabel('Stress Level')
plt.legend()
plt.show()

# Save the final model
model.save('keras_lstm_model_final.h5')
