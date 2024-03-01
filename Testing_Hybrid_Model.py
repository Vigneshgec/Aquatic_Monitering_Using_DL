import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Load the saved models
model_save_path = "C:\\Users\\vigne\\Downloads\\dataset1\\"
autoencoder = load_model(f"{model_save_path}autoencoder_model.h5")
encoder = load_model(f"{model_save_path}encoder_model.h5")
iso_forest = joblib.load(f"{model_save_path}isolation_forest_model.joblib")
scaler = joblib.load(f"{model_save_path}scaler_model.pkl")

# Synthetic new input data
# Note: Adjust the values based on the actual ranges and characteristics of your data
new_data = pd.DataFrame({
    'Temperature (C)': [136, 22, 24],
    'Turbidity(NTU)': [110, 95, 105],
    'Dissolved Oxygen(g/ml)': [5.5, 4.8, 5.0],
    'PH': [98.5, 8.2, 8.3],
    'Ammonia(g/ml)': [0.95, 0.5, 0.48],
    'Nitrate(g/ml)': [190, 185, 195],
    'Population': [50, 50, 50],
    'Fish_Length(cm)': [7.1, 7.2, 7.15],
    'Fish_Weight(g)': [2.9, 2.95, 3.0]
})

# Original numeric columns used during the training
original_numeric_cols = ['Temperature (C)', 'Turbidity(NTU)', 'Dissolved Oxygen(g/ml)', 'PH', 
                         'Ammonia(g/ml)', 'Nitrate(g/ml)', 'Population', 'Fish_Length(cm)', 'Fish_Weight(g)']

# Scale the original numeric data only
data_scaled = scaler.transform(new_data[original_numeric_cols])

# Encode the new data
encoded_new_data = encoder.predict(data_scaled)

# Predict anomalies
anomalies = iso_forest.predict(encoded_new_data)
anomalies = np.where(anomalies == 1, 0, 1)  # Convert anomaly labels

# Evaluate and print results
for i, anomaly in enumerate(anomalies):
    print(f"Data point {i} is {'faulty (anomaly)' if anomaly == 1 else 'normal'}.")