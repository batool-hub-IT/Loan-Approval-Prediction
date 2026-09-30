from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('loan_approval_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            'no_of_dependents',
            'education',
            'self_employed',
            'income_annum',
            'loan_amount',
            'loan_term',
            'cibil_score',
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ]

        input_data = pd.DataFrame(
            [[data[feature] for feature in features]],
            columns=features
        )

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        result = 'Approved' if prediction == 1 else 'Rejected'

        return jsonify({
            'prediction': int(prediction),
            'result': result,
            'approval_probability': round(float(probability), 2)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8502, debug=False)