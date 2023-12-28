from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the standard scaler and random forest model
scaler = joblib.load(r'D:/DS PROJECT FILES/Models/scaler.sav')
model = joblib.load(r'D:/DS PROJECT FILES/Models/random_forest_grid.sav')

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    # Create a numpy array with the input features
    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    # Use the loaded scaler to transform the input features
    X_std = scaler.transform(X)

    # Make predictions using the loaded random forest model
    Y_pred = model.predict(X_std)

    # Assuming Y_pred is a numerical value
    prediction = float(Y_pred)

    # Render the result template and pass the prediction value
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
