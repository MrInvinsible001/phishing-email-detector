import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/phishing_email_subset.csv")

# Remove rows with missing values
df = df.dropna(subset=["text", "label"])

# Inputs and answers
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

# Convert email text into numbers
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

# Create model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train_tfidf, y_train)

# Predict
predictions = model.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification report:")
print(classification_report(y_test, predictions))