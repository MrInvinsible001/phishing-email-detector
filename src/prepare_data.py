import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/phishing_email_subset.csv")

# Remove rows where email or label is missing
df = df.dropna(subset=["text", "label"])

# X = email text
X = df["text"]

# y = correct answer
y = df["label"]

print("Number of valid emails:", len(X))
print("Number of labels:", len(y))

print("\nFirst email:")
print(X.iloc[0])

print("\nFirst label:")
print(y.iloc[0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))