import data
import matplotlib.pyplot as plt

data_frame = data.load_data()

# This code will create a folder called graphs and save the graphs in that folder
continuos_columns = [col for col in data_frame.columns if data_frame[col].dtype != object]
categorical_columns = [col for col in data_frame.columns if data_frame[col].dtype == object]

for col in continuos_columns:
    plt.hist(data_frame[col])
    plt.title(col.capitalize())
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(f'graphs/{col}.png')

for col in categorical_columns:
    data_frame[col].value_counts().plot(kind='bar')
    plt.title(col.capitalize())
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(f'graphs/{col}.png')

