import pandas as pd, numpy as np
import tensorflow as tf
from .features import extract_url_features, feature_names

def predict_urls(model_path: str, urls: list[str]):
    model = tf.keras.models.load_model(model_path)
    rows = []
    for u in urls:
        feats = extract_url_features(u)
        rows.append([feats[c] for c in feature_names()])
    X = np.array(rows, dtype="float32")
    proba = model.predict(X).ravel()
    return list(zip(urls, proba, ["scam" if p>=0.5 else "benign" for p in proba]))
