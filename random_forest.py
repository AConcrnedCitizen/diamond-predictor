import data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

data_frame = data.load_data()
data_frame = pd.get_dummies(data_frame)

X = data_frame.drop(columns=['price']).values
predictors = data_frame.drop(columns=['price']).columns
y = data_frame['price'].values
# The next line is commented out because this script is configured to train on the entire dataset. Uncomment to split it
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train = X, y # Comment this line to split


rand_forest = RandomForestRegressor().fit(X_train, y_train)
# predictions = rand_forest.predict(X_test) # Uncomment this line to split the data
predictions = rand_forest.predict(X) # Comment this line to split the data

accuracy = data.accuracy(rand_forest, X, y)
print('Accuracy:', accuracy)

# Saving the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rand_forest, model_file)
    model_file.close()

