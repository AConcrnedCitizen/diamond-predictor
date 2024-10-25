import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import numpy as np

# Load and preprocess the data
data_frame = data.load_data()
data_frame = pd.get_dummies(data_frame)

data_frame = data_frame.replace([np.inf, -np.inf], np.nan).dropna()

# Drop the price column from the dataframe
X = data_frame.drop(columns=['price']).values.astype(np.float32)
y = data_frame['price'].values.astype(np.float32)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu')) # Input layer
model.add(Dense(64, activation='relu')) # Hidden layer
model.add(Dense(32, activation='relu')) # Hidden layer
model.add(Dense(1, activation='linear')) # Output layer

# Train the model so that the negative mean squared error is minimised through the loss function
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# Define a function to calculate accuracy within a tolerance
def calculate_accuracy_within_tolerance(y_true, y_pred, tolerance=0.1):
    # Calculate the percentage error
    percentage_error = np.abs((y_true - y_pred) / y_true)
    # Calculate the percentage of predictions within the specified tolerance
    accuracy = np.mean(percentage_error <= tolerance) * 100
    return accuracy

model.save('model.h5')

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42) # Create a KFold object

accuracy_scores = []

for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred = model.predict(X_test).flatten()    
    accuracy = calculate_accuracy_within_tolerance(y_test, y_pred, tolerance=0.1)
    
    accuracy_scores.append(accuracy)

average_accuracy = np.mean(accuracy_scores)
print(f'Average Accuracy across {n_folds} folds: {average_accuracy:.2f}%')
