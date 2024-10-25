import data
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle

# Load the dataframe
data_frame = data.load_data()
data_frame = pd.get_dummies(data_frame)

# Drop the price column from the dataframe
X = data_frame.drop(columns=['price']).values
predictors = data_frame.drop(columns=['price']).columns
y = data_frame['price'].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model using the training sets
svm_regress = SVR().fit(X_train, y_train)
predictions = svm_regress.predict(X_test)

accuracy = data.accuracy(svm_regress, X, y)
print('Accuracy:', accuracy)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(svm_regress, model_file)
    model_file.close()

