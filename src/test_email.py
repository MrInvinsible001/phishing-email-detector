import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("data/phishing_email_subset.csv")

# Remove missing values
df = df.dropna(subset=["text", "label"])

# Inputs and labels
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text to numbers
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Our own email
email = """
URGENT: Your bank account has been suspended.
Click here immediately to verify your account and avoid permanent closure.
"""

# Convert our email using the SAME TF-IDF vocabulary
email_tfidf = vectorizer.transform([email])

# Predict
prediction = model.predict(email_tfidf)[0]

if prediction == 1:
    print("PHISHING")
else:
    print("SAFE")