import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "C:\\Users\\vigne\\Downloads\\dataset1\\IoTpond1.csv"
data = pd.read_csv(file_path)

# Preprocessing
# Select numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns
data_numeric = data[numeric_cols]

# Handle missing values explicitly
data_numeric = data_numeric.fillna(data_numeric.median())

# Check for and remove infinite values
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
data_numeric = data_numeric.dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Variational Autoencoder (VAE) setup
input_dim = X_train.shape[1]
intermediate_dim = 64
latent_dim = 2

inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

decoder_h = layers.Dense(intermediate_dim, activation='relu')
decoder_mean = layers.Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = models.Model(inputs, x_decoded_mean)

reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean)) * input_dim
kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train
vae.fit(X_train, X_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, X_test))

# Reconstruction error for evaluation
predictions = vae.predict(X_test)
mse = np.mean(np.square(X_test - predictions), axis=1)
mae = np.mean(np.abs(X_test - predictions), axis=1)

print(f"Mean Squared Error: {np.mean(mse)}")
print(f"Mean Absolute Error: {np.mean(mae)}")