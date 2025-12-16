import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow


def build_text(df: pd.DataFrame) -> list[str]:
    return (
        "PROMPT: " + df["prompt"].fillna("").astype(str)
        + " RESPONSE: " + df["response"].fillna("").astype(str)
    ).tolist()


def threshold_search(y_true: np.ndarray, scores: np.ndarray, thr_steps: int):
    t_min, t_max = float(np.min(scores)), float(np.max(scores))
    thresholds = np.linspace(t_min, t_max, max(2, int(thr_steps)))

    best_thr = float(thresholds[0])
    best_f1 = -1.0
    best_acc = 0.0

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_acc = float(accuracy_score(y_true, y_pred))
            best_thr = float(thr)

    return best_acc, best_f1, best_thr


@torch.no_grad()
def eval_transformer_min(
    df: pd.DataFrame,
    model_dir: Path,
    max_length: int,
    batch_size: int,
    threshold: float,
    thr_steps: int,
    search_threshold: bool
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    y_true = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values
    texts = build_text(df)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    probs = []
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
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(class=1)
        probs.append(p)

    y_prob = np.concatenate(probs)

    if search_threshold:
        best_acc, best_f1, best_thr = threshold_search(y_true, y_prob, thr_steps)
    else:
        best_thr = float(threshold)
        y_pred = (y_prob >= best_thr).astype(int)
        best_acc = float(accuracy_score(y_true, y_pred))
        best_f1 = float(f1_score(y_true, y_pred))

    metrics = {
        "eval_accuracy": float(best_acc),
        "eval_f1": float(best_f1),
        "best_threshold": float(best_thr),
    }

    try:
        metrics["eval_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["eval_auc"] = None

    return metrics


def eval_cross_min(
    df: pd.DataFrame,
    model_dir: Path,
    batch_size: int,
    threshold: float,
    thr_steps: int,
    search_threshold: bool
):
    from sentence_transformers import CrossEncoder

    y_true = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values
    pairs = list(
        zip(
            df["prompt"].fillna("").astype(str).tolist(),
            df["response"].fillna("").astype(str).tolist(),
        )
    )

    ce = CrossEncoder(str(model_dir), num_labels=1)

    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        s = ce.predict(chunk)
        scores.append(np.array(s))

    y_score = np.concatenate(scores)

    if search_threshold:
        best_acc, best_f1, best_thr = threshold_search(y_true, y_score, thr_steps)
    else:
        best_thr = float(threshold)
        y_pred = (y_score >= best_thr).astype(int)
        best_acc = float(accuracy_score(y_true, y_pred))
        best_f1 = float(f1_score(y_true, y_pred))

    metrics = {
        "eval_accuracy": float(best_acc),
        "eval_f1": float(best_f1),
        "best_threshold": float(best_thr),
    }

    try:
        metrics["eval_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["eval_auc"] = None

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["transformer", "cross"], required=True)
    ap.add_argument("--val_csv", default="data/processed/pairwise_val.csv")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--experiment", default="llm-mlops")
    ap.add_argument("--source_run_id", required=True)

    # minimal knobs
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)

    # threshold search (tiny + cheap)
    ap.add_argument("--thr_steps", type=int, default=25)
    ap.add_argument("--search_threshold", action="store_true", help="enable tiny threshold search")

    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    for col in ["prompt", "response", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.val_csv}")

    if args.n > 0 and len(df) > args.n:
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"eval_min_{args.model_type}"):
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("source_run_id", args.source_run_id)
        mlflow.log_param("n_eval", int(len(df)))
        mlflow.log_param("batch_size", int(args.batch_size))
        mlflow.log_param("threshold", float(args.threshold))
        mlflow.log_param("thr_steps", int(args.thr_steps))
        mlflow.log_param("search_threshold", bool(args.search_threshold))

        if args.model_type == "transformer":
            mlflow.set_tag("model_type", "transformer_classifier")
            mlflow.log_param("max_length", int(args.max_length))
            res = eval_transformer_min(
                df, model_dir, args.max_length, args.batch_size,
                args.threshold, args.thr_steps, args.search_threshold
            )
        else:
            mlflow.set_tag("model_type", "cross_encoder")
            res = eval_cross_min(
                df, model_dir, args.batch_size,
                args.threshold, args.thr_steps, args.search_threshold
            )

        mlflow.log_metric("eval_f1", float(res["eval_f1"]))
        mlflow.log_metric("eval_accuracy", float(res["eval_accuracy"]))
        mlflow.log_metric("best_threshold", float(res["best_threshold"]))
        if res.get("eval_auc") is not None:
            mlflow.log_metric("eval_auc", float(res["eval_auc"]))

        print("✅ Minimal eval done on", len(df), "samples")
        print(res)


if __name__ == "__main__":
    main()
