import joblib

from email import policy
from email.parser import BytesParser


# Load trained model
model = joblib.load("models/phishing_email_model.joblib")


def extract_eml_text(file_path):
    """Extract subject and readable text from an .eml file."""

    with open(file_path, "rb") as file:
        message = BytesParser(policy=policy.default).parse(file)

    subject = message.get("subject", "")

    plain_parts = []
    html_parts = []

    if message.is_multipart():
        for part in message.walk():

            if part.get_content_disposition() == "attachment":
                continue

            content_type = part.get_content_type()

            if content_type == "text/plain":
                plain_parts.append(part.get_content())

            elif content_type == "text/html":
                html_parts.append(part.get_content())

    else:
        content_type = message.get_content_type()

        if content_type == "text/plain":
            plain_parts.append(message.get_content())

        elif content_type == "text/html":
            html_parts.append(message.get_content())

    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_parts:
        body = "\n".join(html_parts)
    else:
        body = ""

    return f"Subject: {subject}\n\n{body}"


def get_pasted_email():
    """Read a multi-line email from the terminal."""

    print("Paste the email below.")
    print("Type END on its own line when finished.")
    print()

    lines = []

    while True:
        line = input()

        if line == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def explain_prediction(email):
    """Show features that contributed most to the prediction."""

    feature_union = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    # Transform the email using the exact trained feature pipeline
    transformed_email = feature_union.transform([email])

    # Get feature names
    feature_names = feature_union.get_feature_names_out()

    # Logistic Regression coefficients
    coefficients = classifier.coef_[0]

    # Contribution of each feature to this particular email
    contributions = transformed_email.toarray()[0] * coefficients

    # Features pushing toward phishing
    phishing_indices = contributions.argsort()[-10:][::-1]

    print()
    print("========== TOP PHISHING SIGNALS ==========")

    shown = 0

    for index in phishing_indices:
        value = contributions[index]

        if value <= 0:
            continue

        print(f"{feature_names[index]}  ({value:.4f})")
        shown += 1

        if shown == 10:
            break

    # Features pushing toward safe
    safe_indices = contributions.argsort()[:10]

    print()
    print("========== TOP SAFE SIGNALS ==========")

    shown = 0

    for index in safe_indices:
        value = contributions[index]

        if value >= 0:
            continue

        print(f"{feature_names[index]}  ({value:.4f})")
        shown += 1

        if shown == 10:
            break


# Choose input method
print("1. Paste email")
print("2. Load .eml file")
print()

choice = input("Choose 1 or 2: ").strip()

if choice == "1":

    email = get_pasted_email()

elif choice == "2":

    file_path = input("Enter .eml file path: ").strip().strip('"')

    if not file_path.lower().endswith(".eml"):
        print("Error: option 2 requires an .eml file.")
        raise SystemExit

    try:
        email = extract_eml_text(file_path)

    except FileNotFoundError:
        print("File not found.")
        raise SystemExit

    except Exception as error:
        print("Could not read the email file.")
        print("Error:", error)
        raise SystemExit

else:

    print("Invalid choice.")
    raise SystemExit


# Predict
prediction = model.predict([email])[0]

probabilities = model.predict_proba([email])[0]

safe_probability = probabilities[0]
phishing_probability = probabilities[1]


print()
print("Prediction:", "PHISHING" if prediction == 1 else "SAFE")
print(f"Safe probability: {safe_probability:.2%}")
print(f"Phishing probability: {phishing_probability:.2%}")

explain_prediction(email)