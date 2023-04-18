import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load the data
train_data = pd.read_csv('train_features.csv')
test_data = pd.read_csv('test_features.csv')

# Normalize the data
train_mean = train_data.mean()
train_std = train_data.std()
train_data = (train_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001))

# Train the model
model.fit(train_data, epochs=100, verbose=0)

# Predict the anomaly scores for the test data
test_scores = model.predict(test_data)

# Save the results to a CSV file
np.savetxt('test_scores.csv', test_scores, delimiter=',')
