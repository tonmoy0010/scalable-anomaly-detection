import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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

# Split the training data into multiple subsets
train_X_list = np.array_split(train_X, 5)
train_y_list = np.array_split(train_y, 5)

# Define the ensemble of classifiers
models = []
for i in range(5):
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
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models.append(cnn_model)
    
    rnn_model = Sequential([
        Reshape((train_X.shape[1], 1), input_shape=(train_X.shape[1], 1)),
        LSTM(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    models.append(rnn_model)

# Train each model on a different subset of the training data
for i, model in enumerate(models):
    X = train_X_list[i % 5]
    y = train_y_list[i % 5]
    model.fit(X, y, epochs=10, batch_size=32)

# Evaluate the performance of the ensemble on the test data
cnn_output = cnn_model.predict(test_X)
rnn_output = rnn_model.predict(test_X)
ensemble_output = (cnn_output + rnn_output) / 2
threshold = 0.5
anomalies = (ensemble_output > threshold).astype(int)

precision = precision_score(test_y, anomalies)
recall = recall_score(test_y, anomalies)
f1 = f1_score(test_y, anomalies)
auc_roc = roc_auc_score(test_y, ensemble_output)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC-ROC: {auc_roc}")

# Print the number of anomalies detected
num_anomalies = np.sum(anomalies)
print(f"Number of anomalies detected: {num_anomalies}")
