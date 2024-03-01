import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
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

# Split the dataset
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# One-Class SVM model setup
model = OneClassSVM(kernel='rbf', gamma='auto').fit(X_train)

# Predict and evaluate
preds = model.predict(X_test)
# The model returns -1 for outliers and 1 for inliers. Convert to a binary representation where 1=outlier, 0=inlier for MSE calculation.
binary_preds = np.where(preds == -1, 1, 0)

# Normally, MSE and MAE are not directly applicable for One-Class SVM anomaly detection evaluation,
# since it's unsupervised and we don't have true anomaly labels. However, for demonstration purposes,
# assuming binary_preds against a dummy zero array might give us a sense of deviation.
mse = mean_squared_error(np.zeros_like(binary_preds), binary_preds)
mae = mean_absolute_error(np.zeros_like(binary_preds), binary_preds)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')