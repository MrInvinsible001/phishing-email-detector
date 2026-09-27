import joblib


# Load trained model
model = joblib.load("models/phishing_email_model.joblib")


# Email to classify
email = """
URGENT: Your bank account has been suspended.
Click here immediately to verify your account and avoid permanent closure.
"""


# Predict class
prediction = model.predict([email])[0]

# Get probabilities
probabilities = model.predict_proba([email])[0]

safe_probability = probabilities[0]
phishing_probability = probabilities[1]


print("Prediction:", "PHISHING" if prediction == 1 else "SAFE")
print(f"Safe probability: {safe_probability:.2%}")
print(f"Phishing probability: {phishing_probability:.2%}")