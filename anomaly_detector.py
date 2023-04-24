import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, LSTM, RepeatVector, TimeDistributed

from sklearn.model_selection import train_test_split

# Load the extracted features from the CSV files
train_df = pd.read_csv('train_features.csv')
test_df = pd.read_csv('test_features.csv')

# Convert the features and labels to numpy arrays
train_X = np.array(train_df.drop(['0'], axis=1))
train_y = np.array(train_df['0'])
test_X = np.array(test_df.drop(['0'], axis=1))
test_y = np.array(test_df['0'])

# Add an extra dimension to the input
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# Split the training data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# Define the CNN model architecture
cnn_model = Sequential()
cnn_model.add(Conv1D(32, 3, activation='relu', input_shape=(train_X.shape[1], 1)))
cnn_model.add(Reshape((64,)))
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Reshape((64, 1)))
cnn_model.add(Conv1D(64, 3, activation='relu', padding='same'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))

# Define the RNN model architecture
# Define the RNN model architecture
rnn_model = Sequential()
rnn_model.add(LSTM(units=64, input_shape=(train_X.shape[1], train_X.shape[2])))
rnn_model.add(Dropout(rate=0.2))
rnn_model.add(RepeatVector(train_X.shape[1]))
rnn_model.add(LSTM(units=64, return_sequences=True))
rnn_model.add(Dropout(rate=0.2))
rnn_model.add(TimeDistributed(Dense(units=train_X.shape[2])))
rnn_model.add(Reshape((train_X.shape[1]//2, train_X.shape[2]*2)))

# Compile the CNN and RNN models
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train each model on a different subset of the training data
cnn_model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=10, batch_size=32)
rnn_model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=10, batch_size=32)


# Evaluate the CNN and RNN models on the test data
cnn_loss, cnn_acc = cnn_model.evaluate(test_X, test_y)
rnn_loss, rnn_acc = rnn_model.evaluate(test_X, test_y)

# Combine the outputs of the CNN and RNN models using simple averaging
cnn_output = cnn_model.predict(test_X)
rnn_output = rnn_model.predict(test_X)
ensemble_output = (cnn_output + rnn_output) / 2

# Use a threshold to determine which instances are anomalous
threshold = 0.5
anomalies = (ensemble_output > threshold).astype(int)

# Print the accuracy and confusion matrix of the ensemble
accuracy = np.mean(anomalies == test_y)
confusion_matrix = tf.math.confusion_matrix(test_y, anomalies).numpy()
print(f"Accuracy: {accuracy}")
print(f"Confusion matrix: {confusion_matrix}")

# Print the number of anomalies detected
num_anomalies = np.sum(anomalies)
print(f"Number of anomalies detected: {num_anomalies}")

# Print the indices of the anomalous instances
anomalous_indices = np.where(anomalies == 1)[0]
print(f"Anomalous instances indices: {anomalous_indices}")

# Print the probability of being anomalous for each instance
anomaly_probabilities = ensemble_output[anomalous_indices]
print(f"Anomaly probabilities: {anomaly_probabilities}")

