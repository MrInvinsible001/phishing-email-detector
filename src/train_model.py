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
from scipy.sparse import hstack

# Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)


# Character-level TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)


# Combine word + character features
X_train_tfidf = hstack([
    X_train_word,
    X_train_char
])

X_test_tfidf = hstack([
    X_test_word,
    X_test_char
])

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
results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": predictions
})

false_positives = results[
    (results["actual"] == 0) & (results["predicted"] == 1)
]

false_negatives = results[
    (results["actual"] == 1) & (results["predicted"] == 0)
]

print("\n========== FALSE POSITIVES ==========")
for _, row in false_positives.head(5).iterrows():
    print("\n", row["text"][:700].replace("\n", " "))

print("\n========== FALSE NEGATIVES ==========")
for _, row in false_negatives.head(5).iterrows():
    print("\n", row["text"][:700].replace("\n", " "))