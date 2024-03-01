import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load and preprocess the dataset
file_path = "C:\\Users\\vigne\\Downloads\\dataset1\\IoTpond1.csv"
data = pd.read_csv(file_path)

# Handle missing values only for numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Advanced Feature Engineering
data['Temp_Oxygen_Ratio'] = data['Temperature (C)'] / (data['Dissolved Oxygen(g/ml)'] + 1)
data['Temp_Turbidity_Interaction'] = data['Temperature (C)'] * data['Turbidity(NTU)']

# Ensure no missing values remain (for numeric data)
data.fillna(method='bfill', inplace=True)

# Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_cols])  # Scale only numeric columns

# Split the dataset
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Define the enhanced autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Sequential([
    Dense(64, activation=LeakyReLU(alpha=0.1), activity_regularizer=l1_l2(l1=0.0001, l2=0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation=LeakyReLU(alpha=0.1))
])

decoder = Sequential([
    Dense(32, activation=LeakyReLU(alpha=0.1)),
    Dense(64, activation=LeakyReLU(alpha=0.1)),
    BatchNormalization(),
    Dense(input_dim, activation="sigmoid")
])

autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)))
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Save the models
model_save_path = "C:\\Users\\vigne\\Downloads\\dataset1\\"
autoencoder.save(f"{model_save_path}autoencoder_model.h5")
encoder.save(f"{model_save_path}encoder_model.h5")

# Save the StandardScaler model
scaler_save_path = f"{model_save_path}scaler_model.pkl"
joblib.dump(scaler, scaler_save_path)

# Encode the data
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)

# Train the Isolation Forest
iso_forest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
iso_forest.fit(encoded_train)

# Detect anomalies
anomalies = iso_forest.predict(encoded_test)
anomalies = np.where(anomalies == 1, 0, 1)  # Convert anomaly labels

# Evaluate the model
pseudo_mse = mean_squared_error(np.zeros_like(anomalies), anomalies)
pseudo_mae = mean_absolute_error(np.zeros_like(anomalies), anomalies)

print(f"Enhanced Hybrid Model - Pseudo Mean Squared Error: {pseudo_mse}")

# Save the Isolation Forest model
joblib.dump(iso_forest, f"{model_save_path}isolation_forest_model.joblib")