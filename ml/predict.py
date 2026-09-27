import os
import joblib
import pandas as pd
import numpy as np

def load_artifacts():
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/packaging_model.joblib')
    return preprocessor, model

def predict_packaging_compatibility(user_input):
    """
    Predicts packaging compatibility based on user input.
    """
    preprocessor, model = load_artifacts()

    # Required keys based on ML features
    required_keys = [
        'initial_moisture', 'final_moisture', 'thickness_micron',
        'wvtr', 'otr', 'packaging_material', 'evidence_quality'
    ]

    # Validate missing inputs
    for key in required_keys:
        if key not in user_input or pd.isna(user_input[key]) or user_input[key] == '':
            raise ValueError(f"Missing required input: {key}")

    # Convert and validate numeric bounds
    try:
        init_m = float(user_input['initial_moisture'])
        fin_m = float(user_input['final_moisture'])
        thick = float(user_input['thickness_micron'])
        wvtr_val = float(user_input['wvtr'])
        otr_val = float(user_input['otr'])
    except (ValueError, TypeError):
        raise ValueError("Numeric inputs must be valid numbers.")

    if thick <= 0:
        raise ValueError("Thickness must be > 0")
    if wvtr_val < 0:
        raise ValueError("WVTR must be >= 0")
    if otr_val < 0:
        raise ValueError("OTR must be >= 0")
    if not (0 <= init_m <= 100):
        raise ValueError("Initial moisture must be between 0 and 100")
    if not (0 <= fin_m <= 100):
        raise ValueError("Final moisture must be between 0 and 100")

    # Calculate derived moisture gain
    moisture_gain = fin_m - init_m

    # Build single-row DataFrame
    input_df = pd.DataFrame([{
        'initial_moisture': float(user_input['initial_moisture']),
        'final_moisture': float(user_input['final_moisture']),
        'moisture_gain': moisture_gain,
        'thickness_micron': float(user_input['thickness_micron']),
        'wvtr': float(user_input['wvtr']),
        'otr': float(user_input['otr']),
        'packaging_material': str(user_input['packaging_material']),
        'evidence_quality': float(user_input['evidence_quality'])
    }])

    # Preprocess
    processed_input = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_input)[0]
    probabilities = model.predict_proba(processed_input)[0]

    max_prob = max(probabilities)

    # Map probabilities to class names safely without hardcoding order
    class_probabilities = {
        str(cls): round(float(prob), 3)
        for cls, prob in zip(model.classes_, probabilities)
    }

    # Generate explanation
    explanation = [
        f"Input material {input_df['packaging_material'].iloc[0]} with WVTR {input_df['wvtr'].iloc[0]} was evaluated.",
        f"Biscuit moisture gain is estimated at {moisture_gain:.2f} g/100g.",
        f"Model probability reflects the classifier's estimated probability for the available training labels. It is not experimentally validated real-world packaging success probability."
    ]

    return {
        "recommendation": prediction,
        "probability": round(max_prob, 3),
        "class_probabilities": class_probabilities,
        "predicted_moisture_gain": round(moisture_gain, 3),
        "explanation": explanation
    }

if __name__ == '__main__':
    # Test execution
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    print("Testing prediction service...")
    try:
        res = predict_packaging_compatibility(test_input)
        print("Success! Result:")
        print(res)
    except Exception as e:
        print("Error:", e)
