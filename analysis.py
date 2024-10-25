import data
from scipy.stats import f_oneway
import pandas as pd

# Load the data
data = data.load_data()

# Select the continuous variables
continuous_vars = data.select_dtypes(include=['float64', 'int64'])
continuous_vars.drop(columns=['price'], inplace=True)
# Calculate the correlation between the continuous variables and the target variable
correlation = continuous_vars.corrwith(data['price'], method='pearson')

# print(correlation)

''' Correlation values:
carat    0.924790
depth    0.014074
table    0.140734
x        0.904027
y        0.905075
z        0.900900
'''

# Select the categorical variables
categorical_vars = data.select_dtypes(include=['object'])

anova_results = {}
# Perform the ANOVA test for each categorical variable
for col in categorical_vars:
    groups = [data['price'][categorical_vars[col] == category].values for category in categorical_vars[col].unique()]
    f_stat, p_value = f_oneway(*groups)
    anova_results[col] = {'f': f_stat, 'p': p_value}

anova_df = pd.DataFrame(anova_results).T
# print(anova_df)

''' anova_df:
                  f              p
cut      151.082073  1.103740e-128
color    185.062794  5.245926e-234
clarity  234.112794   0.000000e+00
''' # These are the results of the ANOVA test, all of them are statistically significant
