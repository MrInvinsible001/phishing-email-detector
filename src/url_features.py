from url_analyzer import extract_urls, analyze_url


def extract_email_url_features(email):
    urls = extract_urls(email)

    # No URLs found
    if not urls:
        return {
            "has_url": 0,
            "number_of_urls": 0,
            "max_url_length": 0,
            "has_ip": 0,
            "uses_https": 0,
            "has_at_symbol": 0,
            "max_num_dots": 0,
            "max_num_hyphens": 0,
            "max_num_subdomains": 0,
        }

    # Analyze only valid URLs
    analyzed_urls = []

    for url in urls:
        features = analyze_url(url)

        if features is not None:
            analyzed_urls.append(features)

    # URLs were found, but none could be parsed
    if not analyzed_urls:
        return {
            "has_url": 1,
            "number_of_urls": len(urls),
            "max_url_length": 0,
            "has_ip": 0,
            "uses_https": 0,
            "has_at_symbol": 0,
            "max_num_dots": 0,
            "max_num_hyphens": 0,
            "max_num_subdomains": 0,
        }

    return {
        "has_url": 1,
        "number_of_urls": len(urls),

        "max_url_length": max(
            feature["url_length"] for feature in analyzed_urls
        ),

        "has_ip": int(any(
            feature["has_ip"] for feature in analyzed_urls
        )),

        "uses_https": int(all(
            feature["uses_https"] for feature in analyzed_urls
        )),

        "has_at_symbol": int(any(
            feature["has_at_symbol"] for feature in analyzed_urls
        )),

        "max_num_dots": max(
            feature["num_dots"] for feature in analyzed_urls
        ),

        "max_num_hyphens": max(
            feature["num_hyphens"] for feature in analyzed_urls
        ),

        "max_num_subdomains": max(
            feature["num_subdomains"] for feature in analyzed_urls
        ),
    }