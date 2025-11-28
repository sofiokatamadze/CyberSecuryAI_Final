#!/usr/bin/env python3
"""
task1_cnn_data_and_cnn.py — Self-contained CNN demo with DATA GENERATED + READ in-code.

- Generates a reproducible synthetic network-flow dataset (64 numeric features per flow).
- Prints a small preview table (paste this into your PDF as “the data used”).
- Demonstrates *reading the same data back* via an in-memory CSV (no external files).
- Converts each row to an 8×8 grayscale image and trains a tiny CNN (benign vs malicious).
- Evaluates on a holdout set and prints accuracy.
"""

import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


def generate_flow_dataset(n_samples: int = 200, n_features: int = 64, seed: int = 42):
    """
    Generate a reproducible toy 'network-flow' dataset.
    Returns X (n_samples, n_features) and y (0/1).
    Rule of thumb: higher mass on a subset of 'hot' features => label=1 (malicious).
    """
    rng = np.random.default_rng(seed)
    X = rng.gamma(shape=2.0, scale=1.0, size=(n_samples, n_features)).astype("float32")
    # choose feature indices that drive the 'attack' label
    hot_idx = np.array([1, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    score = X[:, hot_idx].sum(axis=1)
    y = (score > np.percentile(score, 65)).astype("int32")  # ~35% malicious
    return X, y


def dataframe_with_labels(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    cols = [f"f{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df


def to_csv_text(df: pd.DataFrame, max_rows: int = None) -> str:
    """Return a CSV string (optionally truncated to first max_rows for PDF use)."""
    if max_rows is not None:
        df = df.head(max_rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    return csv_buf.getvalue()


def read_from_csv_text(csv_text: str) -> pd.DataFrame:
    """Read the dataset back from an in-memory CSV string (no files)."""
    return pd.read_csv(io.StringIO(csv_text))


def build_cnn(input_shape=(8, 8, 1)):
    """A tiny CNN suitable for 8×8 grayscale images (binary classification)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    # 1) Generate reproducible data
    X, y = generate_flow_dataset(n_samples=200, n_features=64, seed=42)
    df = dataframe_with_labels(X, y)

    # 2) Print a preview table you can paste into the PDF
    print("\n=== DATA PREVIEW (first 8 rows) ===")
    print(df.head(8).round(3).to_string(index=False))

    # 3) Also show CSV text for embedding in the PDF (first 12 rows recommended)
    csv_preview = to_csv_text(df, max_rows=12)
    print("\n=== CSV SNIPPET (first 12 rows) ===")
    print(csv_preview)

    # 4) Demonstrate READING the data back (no files) via in-memory CSV
    df_reloaded = read_from_csv_text(csv_preview)
    # Coerce to numpy and pad back to 64 features if truncated preview was used
    feature_cols = [c for c in df_reloaded.columns if c.startswith("f")]
    X_reload = df_reloaded[feature_cols].values.astype("float32")
    y_reload = df_reloaded["label"].values.astype("int32")
    # (X_reload / y_reload shown just to prove we can read the same schema back)

    # 5) Convert full dataset into 8×8 "images"
    X_full = X.reshape(-1, 8, 8, 1)
    y_full = y

    # 6) Train/test split
    idx = np.arange(len(y_full))
    rng = np.random.default_rng(123)
    rng.shuffle(idx)
    n_train = int(0.8 * len(idx))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    Xtr, Xte = X_full[train_idx], X_full[test_idx]
    ytr, yte = y_full[train_idx], y_full[test_idx]

    # 7) Build & train CNN
    model = build_cnn(input_shape=(8, 8, 1))
    _ = model.fit(Xtr, ytr, epochs=10, batch_size=16, validation_split=0.2, verbose=0)

    # 8) Evaluate
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest Accuracy = {acc:.3f} (loss={loss:.3f})")

    # 9) Notes to include in your PDF
    print("\nNotes:")
    print("- 64 per-flow numerical features are arranged as 8×8 grayscale images.")
    print("- Labels designate flows with high mass on selected features as malicious (toy rule).")
    print("- The dataset is embedded in the script (seed=42) and reloaded from in-memory CSV for demo.")


if __name__ == "__main__":
    # Make TF a bit quieter
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
