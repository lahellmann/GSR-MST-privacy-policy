import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
        labels.append(target[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Parameters
sequence_length = 60  # Use last 60 minutes to predict the next minute's stress level

# Create sequences
X, y = create_sequences(scaled_features, target.values, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save X_test and y_test to .npy files
np.save('Keras_X_test.npy', X_test)
np.save('Keras_y_test.npy', y_test)


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Stress Levels')
plt.xlabel('Sample Index')
plt.ylabel('Stress Level')
plt.legend()
plt.show()

# Save the model
model.save('lstm_model.h5')


#evaluation
# Evaluate the Keras model
keras_test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Keras Test Loss: {keras_test_loss:.4f}')

# Get predictions from the Keras model
keras_predictions = model.predict(X_test)

# Plot actual vs predicted for Keras model
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', alpha=0.5)
plt.plot(keras_predictions, label='Keras Predicted', alpha=0.5)
plt.title('Keras Model: Actual vs Predicted Stress Levels')
plt.xlabel('Sample Index')
plt.ylabel('Stress Level')
plt.legend()
plt.show()
