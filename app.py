import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('accident.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]


    features_name = ['Number_of_vehicles_involved',
                     'Number_of_casualties','Day_of_week','Age_band_of_driver',
                     'Area_accident_occured','Types_of_Junction','Light_conditions',
                     'Weather_conditions','Time_hour']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output[0] == 0:
        res_val = " Fatal injury"
    elif output[0] == 1:
        res_val=" Serious Injury"
    else:
        res_val = " Slight Injury"

    return render_template('index.html', prediction='Person suffered{}'.format(res_val))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)