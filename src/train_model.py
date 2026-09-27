import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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


# Word + character TF-IDF
features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2
        )
    )
])


# Complete pipeline
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("V5: Word + Character TF-IDF")
print()

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

print("\nAccuracy:", accuracy)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["SAFE", "PHISHING"]
    )
)


# Save trained pipeline
model_path = "models/phishing_email_model.joblib"

joblib.dump(model, model_path)

print("Model saved to:", model_path)