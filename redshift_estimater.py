import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_probability as tfp


df = pd.read_csv("SDSS_DR17_label.csv")

selected_columns = ['zWarning', 'redshift', 'psfMag_u', 'modelMag_u', 'cmodelMag_u',
       'extinction_u',  'psfMag_g', 'modelMag_g', 'cmodelMag_g',
       'extinction_g',  'psfMag_r', 'modelMag_r', 'cmodelMag_r',
       'extinction_r',  'psfMag_i', 'modelMag_i', 'cmodelMag_i',
       'extinction_i',  'psfMag_z', 'modelMag_z', 'cmodelMag_z',
       'extinction_z',  'w1', 'w2']

df_select = df[selected_columns]

# Data preprocessing
X = df_select.drop(['zWarning', 'redshift'], axis=1).values
y = df_select['redshift'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Size of X_train:", X_train.shape)
print("Size of y_train:", y_train.shape)
print("Size of X_test:", X_test.shape)
print("Size of y_test:", y_test.shape)

# Normalise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define MDN
class SimpleMDN(tf.keras.Model):
    def __init__(self, num_components):
        super(SimpleMDN, self).__init__()
        self.num_components = num_components
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.mixture_dense_mu = tf.keras.layers.Dense(num_components)
        self.mixture_dense_sigma = tf.keras.layers.Dense(num_components)
        # adding the layer of the mixing coefficient
        self.mixture_dense_pi = tf.keras.layers.Dense(num_components, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mu = self.mixture_dense_mu(x)
        sigma = tf.nn.softplus(self.mixture_dense_sigma(x)) + 1e-5
        pi = self.mixture_dense_pi(x)  # Compuate the mixing coefficient

        # Construct a mixture of Gaussian distributions using mixture coefficients, mean, and variance
        components_distribution = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=pi),
            components_distribution=tfp.distributions.Normal(loc=mu, scale=sigma)
        )
    
        # Calculate the expected value (mean) of a mixture of Gaussian distributions
        mixture_mean = components_distribution.mean()

        return mixture_mean  # Return the mean 

num_components = 3  # the number of mixture components

# Build and compile the model
model = SimpleMDN(num_components)
model.compile(optimizer='adam', loss='mean_squared_error')  # Use MSE as the loss function

import pickle
# Create an empty dictionary to store training history
all_history = {}

# Create a K-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_loss_values = []
best_val_loss = float('inf')  # Initialize the optimal validation loss

# Select the best model
for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Create an empty dictionary to store the current fold’s training history
    fold_history = {"loss": [], "val_loss": []}

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=512, validation_data=(X_val_fold, y_val_fold))

    # Save training history after current fold ends
    fold_history["loss"].extend(history.history["loss"])
    fold_history["val_loss"].extend(history.history["val_loss"])

    # Add the history of the current fold to the overall history
    all_history[f"fold_{fold+1}"] = fold_history

    # Evaluate model performance using validation set
    val_loss = model.evaluate(X_val_fold, y_val_fold)
    print("Validation loss:", val_loss)

    # Save the best performing model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# Finally evaluate the best model on the test set
test_loss = best_model.evaluate(X_test_scaled, y_test)
print("Final test set loss:", test_loss)

from sklearn.preprocessing import StandardScaler

# Create standardized objects
scaler = StandardScaler()

# Standardize the feature matrix
X_scaled = scaler.fit_transform(X)

# Define batch size
batch_size = 1000

# Calculate the total number of batches
num_batches = len(X_scaled) // batch_size
if len(X_scaled) % batch_size != 0:
    num_batches += 1

# Initialize an empty array to store prediction results
z_pred_all = []

# Step by step processing of each batch of data
for i in range(num_batches):
    # Get the data of the current batch
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(X_scaled))
    X_batch = X_scaled[start_idx:end_idx]
    
    # Use the model to predict
    model_output_all = best_model.predict(X_batch)

    # Print the structure of the model output
    print("Model output shape:", model_output_all.shape)

    # Add the results of the current batch to the total results list
    z_pred_all.extend(model_output_all)

# Convert predictions to NumPy array
z_pred_all = np.array(z_pred_all)

# Print the results
print("The prediction of redshift:", z_pred_all)

def save_redshift_to_csv(z_pred, file_path):
    """
    Save predicted redshift values to a CSV file.
  
     :param z_pred: Array of predicted redshift values
     :param file_path: Path to the CSV file to save
    """
    # Create a DataFrame to store predicted redshift values
    # Use an index parameter
    df = pd.DataFrame({'predicted_redshift': z_pred}, index=range(len(z_pred)))

    # Save DataFrame as CSV file
    df.to_csv(file_path, index=False)

# Example
z_pred_all = z_pred_all  # Assume redshift_pred_all is an array containing predicted redshift values

# Specify the path to save the CSV file
file_path = 'z_pred_all.csv'

# Save predicted redshift values as CSV file
save_redshift_to_csv(z_pred_all, file_path)

