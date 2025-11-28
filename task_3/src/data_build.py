import random, csv, os
from .features import extract_url_features, feature_names

_RANDOM_BENIGN = [
    "https://www.wikipedia.org/wiki/Neural_network",
    "https://www.stanford.edu/ai/index.html",
    "https://news.ycombinator.com/item?id=12345",
    "https://www.microsoft.com/en-us/security",
    "https://www.khanacademy.org/math",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://www.georgetown.edu/academics",
    "https://www.un.org/en/climatechange",
    "https://www.python.org/dev/peps/pep-0008/",
    "https://openai.com/research"
]

_RANDOM_SCAM = [
    "http://192.168.0.30/login/verify",
    "http://secure-paypal.com.login-update.co/account/verify",
    "http://bank-secure-update-login.ru/account/urgent",
    "http://bonus-free-money.click/winner/claim",
    "https://verify-update-secure-paypal.com/confirm",
    "http://update-account.paypa1.com/login",
    "http://secure-account.bank.verify-login.cn/account",
    "http://click-bonus-free.win/urgent/reward",
    "http://confirm.update-secure-payments.co/verify",
    "http://account-verify-login-bank.com/update"
]

def generate_synthetic_dataset(n=5000, scam_ratio=0.5):
    k_scam = int(n * scam_ratio)
    k_ben = n - k_scam
    urls = []
    # jitter benign/scam seeds
    for _ in range(k_ben):
        u = random.choice(_RANDOM_BENIGN)
        urls.append((u, 0))
    for _ in range(k_scam):
        u = random.choice(_RANDOM_SCAM)
        # add noise: random subdomain/params
        from urllib.parse import urlparse, urlunparse
        p = urlparse(u)
        host = (f"sub{random.randint(1,9)}." if random.random()<0.4 else "") + (p.hostname or "")
        query = p.query + (("&id=" + str(random.randint(10,9999))) if random.random()<0.5 else "")
        p = p._replace(netloc=host, query=query)
        u = urlunparse(p)
        urls.append((u, 1))
    random.shuffle(urls)
    return urls

def build_or_load_csv(path: str, n=5000, scam_ratio=0.5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        rows = []
        for url, label in generate_synthetic_dataset(n, scam_ratio):
            feats = extract_url_features(url)
            feats["label"] = label
            feats["url"] = url
            rows.append(feats)
        fns = feature_names() + ["label","url"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            for r in rows: w.writerow(r)
    return path
