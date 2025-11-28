#!/usr/bin/env python3
import argparse, os, json
from src.data_build import build_or_load_csv
from src.train_eval import train, evaluate
from src.predict import predict_urls

def main():
    ap = argparse.ArgumentParser(description="Scam URL classifier (4-layer FFN)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", default="data/urls_synthetic.csv")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--model", default="models/url_ffn.keras")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--data", default="data/urls_synthetic.csv")
    p_eval.add_argument("--model", default="models/url_ffn.keras")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", default="models/url_ffn.keras")
    p_pred.add_argument("--urls", nargs="+", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        csv_path = build_or_load_csv(args.data, n=5000, scam_ratio=0.5)
        metrics = train(csv_path, model_out=args.model, epochs=args.epochs)
        print("Training metrics:", json.dumps(metrics, indent=2))

    elif args.cmd == "eval":
        csv_path = build_or_load_csv(args.data, n=5000, scam_ratio=0.5)
        print("Eval metrics:", json.dumps(evaluate(csv_path, args.model), indent=2))

    elif args.cmd == "predict":
        for url, p, cls in predict_urls(args.model, args.urls):
            print(f"{p:0.4f} -> {cls}\t{url}")

if __name__ == "__main__":
    main()
