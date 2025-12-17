# src/evaluate.py
import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow


def dir_size_mb(p: Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def build_text(df: pd.DataFrame) -> list[str]:
    # Backward compatible with your old binary format: prompt + response
    return (
        "PROMPT: " + df["prompt"].fillna("").astype(str)
        + " RESPONSE: " + df["response"].fillna("").astype(str)
    ).tolist()


def threshold_search(y_true: np.ndarray, scores: np.ndarray, steps: int = 300):
    t_min, t_max = float(np.min(scores)), float(np.max(scores))
    thresholds = np.linspace(t_min, t_max, steps)

    best = {"best_threshold": None, "eval_f1": -1.0, "eval_accuracy": None}
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best["eval_f1"]:
            best["eval_f1"] = float(f1)
            best["eval_accuracy"] = float(accuracy_score(y_true, y_pred))
            best["best_threshold"] = float(thr)
    return best


@torch.no_grad()
def eval_transformer(df: pd.DataFrame, model_dir: Path, max_length: int, batch_size: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    y_true = df["label"].astype(int).values
    texts = build_text(df)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    probs = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        logits = model(**inputs).logits
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.append(p)
    dt = time.perf_counter() - t0

    y_score = np.concatenate(probs)
    latency_ms = (dt / max(1, len(texts))) * 1000.0

    best = threshold_search(y_true, y_score)

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = None

    best["eval_auc"] = auc
    best["eval_latency_ms_per_sample"] = float(latency_ms)
    best["eval_model_size_mb"] = float(dir_size_mb(model_dir))
    return best


def eval_cross_encoder(df: pd.DataFrame, model_dir: Path, batch_size: int):
    from sentence_transformers import CrossEncoder

    y_true = df["label"].astype(int).values
    pairs = list(
        zip(
            df["prompt"].fillna("").astype(str).tolist(),
            df["response"].fillna("").astype(str).tolist(),
        )
    )

    ce = CrossEncoder(str(model_dir), num_labels=1)

    scores = []
    t0 = time.perf_counter()
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        s = ce.predict(chunk)
        scores.append(np.array(s))
    dt = time.perf_counter() - t0

    y_score = np.concatenate(scores)
    latency_ms = (dt / max(1, len(pairs))) * 1000.0

    best = threshold_search(y_true, y_score)

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = None

    best["eval_auc"] = auc
    best["eval_latency_ms_per_sample"] = float(latency_ms)
    best["eval_model_size_mb"] = float(dir_size_mb(model_dir))
    return best


def _run_eval(args, df, model_dir):
    if args.model_type == "transformer":
        res = eval_transformer(df, model_dir, args.max_length, args.batch_size)
    else:
        res = eval_cross_encoder(df, model_dir, args.batch_size)

    # Normalize keys to include a single "metric_f1" for selection in promote.py
    out = dict(res)
    out["metric_f1"] = float(res.get("eval_f1", 0.0))
    out["metric_accuracy"] = float(res.get("eval_accuracy", 0.0)) if res.get("eval_accuracy") is not None else None
    out["source_run_id"] = args.source_run_id

    Path("metrics").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("✅ Evaluation done")
    print(out)

    # If running inside MLflow, log metrics/tags too
    if mlflow.active_run() is not None:
        mlflow.set_tag("stage", "evaluation")
        if args.source_run_id:
            mlflow.set_tag("source_run_id", args.source_run_id)

        # log as metrics
        mlflow.log_metric("metric_f1", out["metric_f1"])
        if out["metric_accuracy"] is not None:
            mlflow.log_metric("metric_accuracy", float(out["metric_accuracy"]))
        if out.get("eval_auc") is not None:
            mlflow.log_metric("metric_auc", float(out["eval_auc"]))

        # also tag for quick lookup
        mlflow.set_tag("metric_f1", str(out["metric_f1"]))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_type", choices=["transformer", "cross"], default="transformer")
    ap.add_argument("--source_run_id", default=None)

    ap.add_argument("--val_csv", default="data/processed/mini_val.csv")
    ap.add_argument("--model_dir", required=True)

    ap.add_argument("--experiment", default="llm-mlops")
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--out", default="metrics/scores.json")

    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    model_dir = Path(args.model_dir)

    run_name = args.run_name or f"evaluate_{args.model_type}"

    # If source_run_id provided => log evaluation in MLflow
    if args.source_run_id:
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=run_name):
            _run_eval(args, df, model_dir)
    else:
        # DVC-only
        _run_eval(args, df, model_dir)


if __name__ == "__main__":
    main()
