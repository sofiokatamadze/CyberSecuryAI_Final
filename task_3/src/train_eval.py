import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, confusion_matrix)
from .features import feature_names
from .model_ffn import make_ffn

def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = feature_names()
    X = df[cols].values.astype("float32")
    y = df["label"].values.astype("float32")
    urls = df.get("url", pd.Series([""]*len(df))).values
    return X, y, urls

def train(csv_path: str, model_out: str, epochs=15, seed=42, figs_dir="figs"):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    X, y, _ = load_xy(csv_path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=seed)
    model = make_ffn(X.shape[1])
    # fit Normalization layer on train
    norm = model.layers[1]
    norm.adapt(Xtr)

    cb = [
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    hist = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs,
                     batch_size=64, callbacks=cb, verbose=1)

    # metrics
    p = model.predict(Xte).ravel()
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(yte, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
    auc = roc_auc_score(yte, p)

    # curves
    plt.figure();
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_accuracy_curves.png")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(yte, p)
    plt.figure(); plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "roc_auc.png")); plt.close()

    # Confusion
    cm = confusion_matrix(yte, yhat)
    plt.figure(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                              xticklabels=["benign","scam"], yticklabels=["benign","scam"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "confusion_matrix.png")); plt.close()

    model.save(model_out)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def evaluate(csv_path: str, model_path: str):
    import tensorflow as tf
    X, y, _ = load_xy(csv_path)
    model = tf.keras.models.load_model(model_path)
    p = model.predict(X).ravel()
    yhat = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)[0],
        "recall": precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)[1],
        "f1": precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)[2],
        "auc": roc_auc_score(y, p),
    }
