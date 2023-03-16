import numpy as np
import pandas as pd
import socket

# Function to extract features from log files
def extract_features_from_log(log_file):
    # Load log file data into a pandas dataframe
    log_data = pd.read_csv(log_file)

    # Remove unnecessary columns
    log_data.drop(['timestamp', 'log_level'], axis=1, inplace=True)

    # Convert text data to numerical data
    log_data['message_length'] = log_data['message'].apply(len)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(log_data)

    # Extract features
    features = []
    for i in range(len(scaled_data)):
        feature = []
        for j in range(len(scaled_data[i])):
            if j != 0:
                diff = scaled_data[i][j] - scaled_data[i][j-1]
                feature.append(diff)
        features.append(feature)

    # Create a feature extracted log file
    feature_df = pd.DataFrame(features, columns=log_data.columns[1:])
    feature_df.to_csv(log_file[:-4] + '_feature_extracted.csv', index=False)

    # Return extracted features as numpy array
    return np.array(features)

# Create a socket object
s = socket.socket()

# Get local machine name
host = socket.gethostname()

# Reserve a port for your service.
port = 12345

# Bind to the port
s.bind((host, port))

# Wait for client to connect
s.listen(5)

while True:
    # Wait for client to connect
    c, addr = s.accept()
    print('Got connection from', addr)

    # Receive log file path from client
    log_file_path = c.recv(1024).decode()
    print(f"Received log file path: {log_file_path}")
    
    # Extract features from log file
    features = extract_features_from_log(log_file_path)

    # Send extracted features to anomaly detector
    c.send(features.tobytes())

    # Close the connection
    c.close()
