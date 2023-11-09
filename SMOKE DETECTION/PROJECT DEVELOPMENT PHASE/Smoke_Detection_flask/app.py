from flask import Flask, render_template, request
import numpy as np
import pickle

with open('smoke.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    tvoc = float(request.form['tvoc'])
    raw_h2 = float(request.form['raw_h2'])
    raw_ethanol = float(request.form['raw_ethanol'])
    pressure = float(request.form['pressure'])
    nc0_5 = float(request.form['nc0_5'])
    cnt = float(request.form['cnt'])
    final_features = np.array([temperature, humidity, tvoc, raw_h2, raw_ethanol, pressure, nc0_5, cnt])
    final_features = final_features.reshape(1, -1)
    prediction = model.predict(final_features)[0]

    if prediction == 0:
        prediction_text = 'The input does not indicate smoke detection.'
    else:
        prediction_text = 'The input indicates smoke detection.'
    return render_template('submit.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
