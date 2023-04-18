import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten, Input, LSTM, MaxPooling1D

# Load the KDDCup99 dataset
df = pd.read_csv('Datasets/kddcup99.csv', header=None, index_col=False, dtype={'duration': object})


# Assign column names to the dataframe
col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
             'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
             'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
             'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
             'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
             'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
             'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
df.columns = col_names

# Define attack types to be detected (22 total)
attack_types = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
                'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
                'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.']

# Create a binary label column where 1 indicates the row contains one of the defined attacks and 0 indicates normal traffic
df['attack'] = np.where(df.label.isin(attack_types), 1, 0)

# Drop the original label column
df = df.drop(['label'], axis=1)

# One-hot encode the categorical features
cat_cols = ['protocol_type', 'service', 'flag']
df = pd.get_dummies(df, columns=cat_cols)

# Normalize the data
for col in df.columns:
    if col not in cat_cols and col != 'attack':
        df[col] = (df[col] - df[col].mean()) / df[col].std()

# Convert the dataframe to numpy arrays
X = df.drop(['attack'], axis=1).values
y = df['attack'].values

# Split the data into training and validation sets
train_idx = np.random.choice(range(len(X)), int(len(X) * 0.8), replace=False)
val_idx = np.array(list(set(range(len(X))) - set(train_idx)))
X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]

# Define the model
input_layer = Input(shape=(X_train.shape[1], 1))
conv1_layer = Conv1D(32, 3, activation='relu')(input_layer)
maxpool1_layer = MaxPooling1D(2)(conv1_layer)
conv2_layer = Conv1D(64, 3, activation='relu')(maxpool1_layer)
maxpool2_layer = MaxPooling1D(2)(conv2_layer)
lstm_layer = LSTM(64, return_sequences=True)(maxpool2_layer)
flatten_layer = Flatten()(lstm_layer)
dense1_layer = Dense(64, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense1_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=5, batch_size=128,
                    validation_data=(X_val.reshape(-1, X_train.shape[1], 1), y_val))

# Generate predictions on the test set
y_pred = model.predict(X_test.reshape(-1, X_train.shape[1], 1))

# Create a dataframe with the predicted anomaly scores
anomaly_scores = pd.DataFrame(y_pred, columns=['anomaly_score'])

# Save the dataframe to a CSV file
anomaly_scores.to_csv('anomaly_scores.csv', index=False)

