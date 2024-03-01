import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = "C:\\Users\\vigne\\Downloads\\dataset1\\IoTpond1.csv"
data = pd.read_csv(file_path)

# Preprocessing
# Exclude non-numeric columns before computing the median
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[numeric_cols])

# Split the dataset into training and testing sets
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# Autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim, ))

# Encoder
encoder = Dense(64, activation="relu")(input_layer)
encoder = Dense(32, activation="relu")(encoder)
encoder = Dense(16, activation="relu")(encoder)

# Decoder
decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(decoder)
decoder = Dense(64, activation="relu")(decoder)
decoder = Dense(input_dim, activation='linear')(decoder)

# Model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# Training
history = autoencoder.fit(X_train, X_train,
                          epochs=20,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer, tensorboard]).history

# Prediction on the test set
predictions = autoencoder.predict(X_test)

# Calculate MSE and MAE
mse_value = mean_squared_error(X_test, predictions)
mae_value = mean_absolute_error(X_test, predictions)

print(f"Mean Squared Error: {mse_value}")
print(f"Mean Absolute Error: {mae_value}")