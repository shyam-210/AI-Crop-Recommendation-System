from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = joblib.load('Crop_predictor.joblib')
scaler = joblib.load('MimMaxScaler.joblib')
label_encoder = joblib.load('Crop_LabelEncoder.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    # Get form data
    nitrogen = float(request.form['nitrogen'])
    phosphorous = float(request.form['phosphorous'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare data for prediction
    data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    crop = label_encoder.inverse_transform(prediction)[0]

    # Pass data and prediction to the result page
    return render_template('result.html', nitrogen=nitrogen, phosphorous=phosphorous,
                           potassium=potassium, temperature=temperature, humidity=humidity,
                           ph=ph, rainfall=rainfall, crop=crop)

if __name__ == '__main__':
    app.run(debug=True)
