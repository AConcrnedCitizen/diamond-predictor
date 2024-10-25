from flask import Flask, render_template, request
import pandas as pd
import pickle
import tensorflow as tf


# Create the server object
app = Flask(__name__)

# Load the model file into memory
with open ('model.pkl', 'rb') as file:
    model = pickle.load(file)

# tf.keras.models.load_model('model.h5') # Uncomment this line if you are using a neural network model

# Function to predict the price of a diamond
def price_predict(carat: float, x: float, y: float, z: float, # Carat weight, length, width, and depth of the diamond
                      cut_Fair: bool, cut_Good: bool, cut_Ideal: bool, # The cut of the diamond
                      cut_Premium: bool, cut_Very_Good: bool,
                      color_D: bool, color_E: bool, color_F: bool, # The color of the diamond
                      color_G: bool, color_H: bool, color_I: bool, color_J: bool,
                      clarity_I1: bool, clarity_IF: bool, clarity_SI1: bool, # The clarity of the diamond
                      clarity_SI2: bool, clarity_VS1: bool, clarity_VS2: bool,
                      clarity_VVS1: bool, clarity_VVS2: bool):
    
    # Create a dataframe with the input data
    data = pd.DataFrame({
        'carat': [float(carat)], # Affirming the type of the input data
        'x': [float(x)],
        'y': [float(y)],
        'z': [float(z)],
        'cut_Fair': [bool(cut_Fair)],
        'cut_Good': [bool(cut_Good)],
        'cut_Ideal': [bool(cut_Ideal)],
        'cut_Premium': [bool(cut_Premium)],
        'cut_Very Good': [bool(cut_Very_Good)],
        'color_D': [bool(color_D)],
        'color_E': [bool(color_E)],
        'color_F': [bool(color_F)],
        'color_G': [bool(color_G)],
        'color_H': [bool(color_H)],
        'color_I': [bool(color_I)],
        'color_J': [bool(color_J)],
        'clarity_I1': [bool(clarity_I1)],
        'clarity_IF': [bool(clarity_IF)],
        'clarity_SI1': [bool(clarity_SI1)],
        'clarity_SI2': [bool(clarity_SI2)],
        'clarity_VS1': [bool(clarity_VS1)],
        'clarity_VS2': [bool(clarity_VS2)],
        'clarity_VVS1': [bool(clarity_VVS1)],
        'clarity_VVS2': [bool(clarity_VVS2)]
    })

    
    predictions = model.predict(data) # Make the prediction

    return predictions[0]

# Define the route for the prediction
@app.route('/predict', methods=['POST']) # Forcing the route to only accept POST requests
def predict():
    # Convert the request data into a dictionary
    request_data = request.get_json()
    
    data = { # The other values require more processing but the carat, x, y, and z values can be used as is
        'carat': request_data["carat"],
        'x': request_data["x"],
        'y': request_data["y"],
        'z': request_data["z"],
    }

    # Convert the categorical variables into boolean variables
    cuts = ['Fair', 'Good', 'Ideal', 'Premium', 'Very_Good']
    for cut_type in cuts: # Loop through the cut types
        data[f'cut_{cut_type}'] = request_data["cut"] == cut_type # Check if the cut type is the same as the one in the request data
    
    colours = ['D', 'E', 'F', 'G', 'H', 'I', 'J'] # The different colours of the diamonds, same as the cut types
    for colour_type in colours:
        data[f'color_{colour_type}'] = request_data["colour"] == colour_type

    clarities = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'] # The different clarities of the diamonds, same as the cut types
    for clarity_type in clarities:
        data[f'clarity_{clarity_type}'] = request_data["clarity"] == clarity_type

    # Make the prediction, round it to 2 decimal places and return it
    prediction = float(price_predict(**data)) # Sending it as a **kwargs argument
    return {'price': round(prediction, 2)}

# Define the route for the home page
@app.route('/', methods=['GET']) # Forcing the route to only accept GET requests
def home():
    return render_template('index.html') # Render the index.html template

# If the app is run directly (not imported) then it will run
if __name__ == '__main__':
    app.run(debug=True)