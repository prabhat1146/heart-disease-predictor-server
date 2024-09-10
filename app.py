from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pickle
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='app/templates')
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Load the dictionary of encoders
with open('encoders/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

@app.route('/find-heart-disease', methods=['POST'])
def find_heart_disease():
    try:
        # Parse input data
        data = request.json
        # print(data)
        input_data = pd.DataFrame([data])
        print(input_data.iloc[0])
        input_data = input_data.apply(pd.to_numeric, errors='ignore')

        # Apply LabelEncoder to the categorical columns
        for col in input_data.columns:
            if col in label_encoders:
                # print("en",col)
                le = label_encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col].astype(str))
                except ValueError as e:
                    # print('Unseen label in column ',col)
                    return jsonify({'error': f'Unseen label in column {col}: {e}'}), 400
            else:
                # Handle features without encoders
                # print("ne",col)
                if input_data[col].dtype == 'object':
                    # Handle non-numeric features (e.g., missing encoders)
                    # print('No encoder for column',col)
                    return jsonify({'error': f'No encoder for column: {col}'}), 400

        # Ensure that numerical features are of correct type
        # print(input_data)
        # Predict using the model
        prediction = model.predict(input_data)
        print("Pred: ",prediction)
        probability = model.predict_proba(input_data)[0][1]
        print("Prob: ",probability)

    except Exception as e:
        # Print the error for debugging purposes
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

    return jsonify({
        'prediction': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease',
        'probability': round(probability * 100, 2)  # Convert to percentage
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)


