import re
import ipaddress
from urllib.parse import urlparse


def extract_urls(text):
    pattern = r'''https?://[^\s<>"']+'''
    return re.findall(pattern, text)


def analyze_url(url):
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Check whether hostname is an IP address
    has_ip = False

    try:
        ipaddress.ip_address(hostname)
        has_ip = True
    except ValueError:
        pass

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": hostname.count("."),
        "num_hyphens": hostname.count("-"),
        "num_subdomains": (
            0 if has_ip else max(0, len(hostname.split(".")) - 2)
        ),
        "uses_https": parsed.scheme == "https",
        "has_ip": has_ip,
        "has_at_symbol": "@" in url,
    }

    return features