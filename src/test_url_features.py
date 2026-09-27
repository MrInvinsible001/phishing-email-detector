from url_features import extract_email_url_features


email = """
URGENT: Verify your account.

Visit:
http://example-login.com/verify

More information:
https://example.com/help
"""


features = extract_email_url_features(email)

print("URL features:")

for name, value in features.items():
    print(f"{name}: {value}")