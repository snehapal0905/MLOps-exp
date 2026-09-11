from flask import Flask, request, jsonify
import joblib
import os
app = Flask(__name__)
# Load the trained model artifact
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
@app.route('/', methods=['GET'])
def home():
    return "ML Model Deployment API is Live! Send POST requests to /predict"
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            "error": "Model not found. Train the model first."
        }), 500
    try:
        # Get data from the user request
        data = request.get_json()

        features = data['features']

        # Make a prediction
        prediction = model.predict([features])

        # Return the result
        return jsonify({
            "prediction": int(prediction[0])
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400
if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)