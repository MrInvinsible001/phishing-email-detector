import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from url_features import extract_email_url_features


# -------------------------
# 1. Load dataset
# -------------------------

df = pd.read_csv("data/phishing_email_subset.csv")

df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]


# -------------------------
# 2. Split dataset
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 3. Convert email text to TF-IDF
# -------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -------------------------
# 4. Extract URL features
# -------------------------

train_url_features = [
    extract_email_url_features(email)
    for email in X_train
]

test_url_features = [
    extract_email_url_features(email)
    for email in X_test
]

train_url_df = pd.DataFrame(train_url_features)
test_url_df = pd.DataFrame(test_url_features)


# Convert URL features into sparse matrices
X_train_url = csr_matrix(train_url_df.values)
X_test_url = csr_matrix(test_url_df.values)


# -------------------------
# 5. Combine text + URL features
# -------------------------

X_train_combined = hstack([
    X_train_tfidf,
    X_train_url
])

X_test_combined = hstack([
    X_test_tfidf,
    X_test_url
])


print("TF-IDF training shape:", X_train_tfidf.shape)
print("URL training shape:", X_train_url.shape)
print("Combined training shape:", X_train_combined.shape)


# -------------------------
# 6. Train model
# -------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_combined, y_train)


# -------------------------
# 7. Test model
# -------------------------

predictions = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, predictions)

print("\nV2 Accuracy:", accuracy)

print("\nV2 Classification Report:")
print(classification_report(y_test, predictions))