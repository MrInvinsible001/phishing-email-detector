import pandas as pd

df = pd.read_csv("data/phishing_email_subset.csv")

print("First 5 emails:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset size:")
print(df.shape)

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nOne phishing email:")
print(df[df["label"] == 1]["text"].iloc[0])

print("\nOne safe email:")
print(df[df["label"] == 0]["text"].iloc[0])

print("\nLabel meanings:")
print("0 = Safe")
print("1 = Phishing")