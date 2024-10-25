import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Load data function
def load_data():
    # Load the data by reading the csv file
    data = pd.read_csv('original-dataset.csv')
    data.drop_duplicates() # Drop any possible duplicates

    data = data.drop(columns=['table', 'depth']) # As proven in analysis, these columns have little to no correlation with price
    data = data.drop(columns=['Unnamed: 0']) # Drop the index column
        
    data = data.dropna()# Drop any rows with missing values

    # UNCOMMENT THE FOLLOWING CODE TO REMOVE OUTLIERS THIS IS NOT NEEDED
    # for col in data.columns:
    #     if data[col].dtype != object: # If the column is not a categorical column
    #         # Remove the outliers by keeping only the values that fall within the 5th and 95th percentiles
    #         data = data[(data[col] >= data[col].quantile(0.05)) & (data[col] <= data[col].quantile(0.95))]

    return data


def accuracy(model, X, y):
    # Create an accuracy function that calculates the accuracy of the model
    def accuracy_score(orig, pred):
        MAPE = np.mean(100*(np.abs(orig-pred)/orig))
        return 100 - MAPE
    
    # Create a custom scoring function
    custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
    
    # Return the cross-validated accuracy score
    return cross_val_score(model, X, y, cv=10, scoring=custom_scoring).mean()