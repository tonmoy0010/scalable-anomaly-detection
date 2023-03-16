import socket
import numpy as np
from sklearn.cluster import KMeans

##
# This code uses the socket module to receive the extracted features from the feature extractor, detects anomalies using a clustering algorithm, and logs the anomalies to a file named "anomalies.log".
##

# Function to detect anomalies
def detect_anomalies(features):
    # Apply clustering algorithm to features
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

    # Get cluster labels
    labels = kmeans.labels_

    # Determine anomalies as points in the smaller cluster
    cluster_counts = np.bincount(labels)
    smallest_cluster = np.argmin(cluster_counts)
    anomalies = features[labels == smallest_cluster]

    # Return anomalies as numpy array
    return anomalies

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

    # Receive extracted features from feature extractor
    features = np.frombuffer(c.recv(1024))

    # Detect anomalies in the features
    anomalies = detect_anomalies(features)

    # Log the anomalies
    with open("anomalies.log", "a") as f:
        for anomaly in anomalies:
            f.write(str(anomaly) + "\n")

    # Close the connection
    c.close()
