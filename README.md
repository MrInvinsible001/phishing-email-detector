# Phishing Email Detector

A machine-learning project that classifies email text as SAFE or PHISHING using word-level and character-level TF-IDF features with Logistic Regression.

## How it works

Email -> Word TF-IDF + Character TF-IDF -> Logistic Regression -> SAFE / PHISHING

## Results

Test-set accuracy: 98.0%

5-fold stratified cross-validation:
- Mean accuracy: 98.02%
- Standard deviation: 0.31 percentage points

Ablation:
- Word only: 97.30%
- Character only: 96.12%
- Word + Character: 98.02%

## Setup

Install dependencies:

    pip install -r requirements.txt

Train the model:

    python src/train_model.py

Test an email:

    python src/test_email.py

The tester accepts pasted email text or a .eml file.
For pasted email, type END on its own line when finished.

## Project structure

    phishing-email-detector/
    |- data/
    |- models/
    |- src/
    |  |- train_model.py
    |  |- test_email.py
    |- .gitignore
    |- README.md
    |- requirements.txt

## Limitations

This is a learning and portfolio project, not a production email-security gateway.
Performance on real-world emails can differ from the dataset results.
The displayed probabilities are Logistic Regression model scores, not guaranteed real-world probabilities.

URL-based experiments were evaluated separately but did not improve the final text classifier, so they are not part of the final V5 pipeline.

## Future improvements

- Sender and email-header analysis
- SPF, DKIM and DMARC signals
- Domain reputation
- Probability calibration
- Web interface or API
