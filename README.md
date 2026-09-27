@'
# Phishing Email Detector

A machine-learning project that classifies email text as **SAFE** or **PHISHING** using word-level and character-level TF-IDF features with Logistic Regression.

## How it works

```text
Email
  ↓
Word TF-IDF (1–2 grams)
  +
Character TF-IDF (3–5 grams)
  ↓
Logistic Regression
  ↓
SAFE / PHISHING
  +
Prediction probability
  +
Top contributing features