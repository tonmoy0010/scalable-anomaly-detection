import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, concatenate, Flatten, Reshape
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the KDD Cup 1999 dataset
df = pd.read_csv("Datasets\kddcup99.csv")

# Define the categorical and numerical features
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']

# Label encode the categorical features
for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])

# Standardize the numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define the input shape
input_shape = (None, len(categorical_features) + len(numerical_features))
input_layer_categorical = Input(shape=(len(categorical_features),))
input_layer_numerical = Input(shape=(len(numerical_features),))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

# Convert the labels to categorical format
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train))
y_test = to_categorical(le.transform(y_test))

# Define the CNN layer
input_layer_categorical = Input(shape=(len(categorical_features),))
input_layer_numerical = Input(shape=(len(numerical_features),))
reshape_layer = Reshape((len(numerical_features), 1))(input_layer_numerical)

# Update input_layer_categorical to match the second dimension of reshape_layer
input_layer_categorical_reshaped = Reshape((len(categorical_features), 1))(input_layer_categorical)

input_layer = concatenate([input_layer_categorical_reshaped, reshape_layer], axis=1)
conv1 = Conv1D(filters=16, kernel_size=3, activation='relu')(input_layer)
pool1 = MaxPooling1D(pool_size=1)(conv1)
conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=1)(conv2)
flatten1 = Flatten()(pool2)

# Define the LSTM layer
lstm1 = LSTM(units=32, return_sequences=True)(input_layer)
lstm2 = LSTM(units=64)(lstm1)

# Concatenate the CNN and LSTM layers
concat = concatenate([flatten1, lstm2])

# Define the output layer
output_layer = Dense(units=len(np.unique(df['label'])), activation='softmax')(concat)

# Define the model
model = Model(inputs=[input_layer_categorical, input_layer_numerical], outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x=[X_train[categorical_features], X_train[numerical_features]], y=y_train, validation_data=([X_test[categorical_features], X_test[numerical_features]], y_test), epochs=10, batch_size=64)

# Extract the features using the trained model
feature_extractor = Model(inputs=model.input, outputs=model.layers[3].output)
X_train_features = feature_extractor.predict([X_train[categorical_features], X_train[numerical_features]])
X_test_features = feature_extractor.predict([X_test[categorical_features], X_test[numerical_features]])

# Save the extracted features to a CSV file
pd.DataFrame(X_train_features).to_csv('train_features.csv', index=False)
pd.DataFrame(X_test_features).to_csv('test_features.csv', index=False)
