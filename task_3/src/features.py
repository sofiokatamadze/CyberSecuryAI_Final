import re, math
from urllib.parse import urlparse

_SUSPICIOUS = ["login","verify","update","free","winner","click","bonus",
               "urgent","account","secure","bank","paypal"]

_IPv4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    cnt = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())

def extract_url_features(url: str) -> dict:
    try:
        p = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        p = urlparse("http://" + url)

    host = p.hostname or ""
    path = p.path or ""
    query = p.query or ""

    host_is_ip = bool(_IPv4_RE.match(host))
    num_subdomains = host.count(".")
    s = url

    specials = sum(s.count(ch) for ch in r"!@#$%^&*()_+-=[]{}|;:',.<>/?~`%")
    digits = sum(ch.isdigit() for ch in s)

    feats = {
        "url_len": len(s),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "num_subdomains": num_subdomains,
        "uses_https": 1.0 if p.scheme.lower() == "https" else 0.0,
        "has_ip_in_host": 1.0 if host_is_ip else 0.0,
        "num_digits": digits,
        "num_specials": specials,
        "count_at": s.count("@"),
        "count_dash": s.count("-"),
        "count_underscore": s.count("_"),
        "count_percent": s.count("%"),
        "count_dot": s.count("."),
        "hostname_entropy": shannon_entropy(host),
    }
    # keyword flags
    for w in _SUSPICIOUS:
        feats[f"kw_{w}"] = 1.0 if w in s.lower() else 0.0
    return feats

def feature_names():
    base = ["url_len","host_len","path_len","query_len","num_subdomains",
            "uses_https","has_ip_in_host","num_digits","num_specials",
            "count_at","count_dash","count_underscore","count_percent",
            "count_dot","hostname_entropy"]
    kw = [f"kw_{w}" for w in _SUSPICIOUS]
    return base + kw
