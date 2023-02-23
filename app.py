from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define a route to handle the form submission
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = int(request.form['chas'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = int(request.form['rad'])
        tax = int(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstat = float(request.form['lstat'])

        # Make a prediction using the model
        features = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
        price = model.predict(features)[0]

        # Render the template with the predicted price
        return render_template('result.html', price=price)

    # Render the form template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)