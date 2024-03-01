import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "C:\\Users\\vigne\\Downloads\\dataset1\\IoTpond1.csv"
data = pd.read_csv(file_path)

# Preprocessing
# Select numeric columns only, excluding non-numeric columns for preprocessing
numeric_cols = data.select_dtypes(include=['number']).columns
data_numeric = data[numeric_cols].fillna(data[numeric_cols].median())

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# Split the dataset into training and test sets
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Initialize and fit the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train)

# Predicting on the test set
# The model will predict -1 for outliers and 1 for inliers
predictions = iso_forest.predict(X_test)

# Scores (the lower, the more abnormal)
scores = iso_forest.decision_function(X_test)

# Convert scores to positive values, with higher values indicating more normal
scores_pos = -scores

# Evaluation (Pseudo)
# Since this is unsupervised, we don't have true labels to calculate accuracy.
# Instead, we look at the summary statistics of the scores.
pseudo_mae = mean_absolute_error(scores_pos, np.zeros_like(scores_pos))
pseudo_mse = mean_squared_error(scores_pos, np.zeros_like(scores_pos))

print(f"Pseudo Mean Absolute Error: {pseudo_mae}")
print(f"Pseudo Mean Squared Error: {pseudo_mse}")