import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = "C:\\Users\\vigne\\Downloads\\dataset1\\IoTpond1.csv"
data = pd.read_csv(file_path)

# Preprocessing
numeric_cols = data.select_dtypes(include=['number']).columns
data_numeric = data[numeric_cols]

# Handle missing values
data_numeric.fillna(data_numeric.median(), inplace=True)

# Check for and handle infinite values
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
data_numeric.dropna(inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)
scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

# Split the dataset
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# LSTM model setup
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(X_train.shape[2])
])

model.compile(optimizer='adam', loss='mae')

# Train the model
history = model.fit(X_train, X_train, epochs=20, batch_size=32, validation_data=(X_test, X_test), verbose=2)

# Prediction and evaluation
predicted = model.predict(X_test)
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[2]))

mse = mean_squared_error(X_test_reshaped, predicted, multioutput='raw_values')
mae = mean_absolute_error(X_test_reshaped, predicted, multioutput='raw_values')

print(f'Mean Squared Error: {np.mean(mse)}')
print(f'Mean Absolute Error: {np.mean(mae)}')