import pandas as pd

from url_analyzer import extract_urls, analyze_url


# Load dataset
df = pd.read_csv("data/phishing_email_subset.csv")

# Remove missing emails
df = df.dropna(subset=["text", "label"])

found = 0

# Search through the dataset
for index, row in df.iterrows():

    email = row["text"]
    label = row["label"]

    urls = extract_urls(email)

    # Only show emails that contain URLs
    if urls:
        print("\n" + "=" * 60)
        print("Email number:", index)
        print("Label:", "PHISHING" if label == 1 else "SAFE")
        print("URLs found:", len(urls))

        for url in urls:
            print("\nURL:", url)

            features = analyze_url(url)

            for name, value in features.items():
                print(f"{name}: {value}")

        found += 1

    # Stop after finding 5 emails with URLs
    if found == 5:
        break

print("\nFound", found, "emails containing detectable URLs.")