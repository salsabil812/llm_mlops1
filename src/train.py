# src/train.py
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder, InputExample
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

import mlflow
from mlflow.tracking import MlflowClient
from mlflow import pyfunc
from mlflow import transformers as mlt

# -------------------------
# ENV / UTILS
# -------------------------
def set_env_no_wandb():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _validate_df(df: pd.DataFrame, name: str):
    required = {"prompt", "response", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {list(df.columns)}")


def _clean_text_cols(df: pd.DataFrame):
    df = df.copy()
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(float)
    return df


def build_hf_dataset(df: pd.DataFrame):
    df = df.copy()
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    texts = ("PROMPT: " + df["prompt"] + " RESPONSE: " + df["response"]).tolist()
    labels = df["label"].tolist()
    return Dataset.from_dict({"text": texts, "label": labels})


# -------------------------
# CROSS ENCODER (OPTIONAL)
# -------------------------
def train_cross_encoder(
    train_df, val_df, out_dir: Path,
    model_name: str, batch_size: int, epochs: int,
    fast_ci: bool = False,
):
    """
    Pour deadline/CI: conseille de le SKIP.
    Si tu l'actives: on coupe dur en fast_ci (epochs=1) mais ça peut rester lent sur CPU.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _clean_text_cols(train_df)
    val_df = _clean_text_cols(val_df)

    if fast_ci:
        epochs = 1

    train_samples = [
        InputExample(texts=[r.prompt, r.response], label=float(r.label))
        for r in train_df.itertuples()
    ]

    cross_encoder = CrossEncoder(model_name, num_labels=1)

    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=cross_encoder.smart_batching_collate,
    )

    total_steps = max(1, math.ceil(len(train_samples) / batch_size) * epochs)
    warmup_steps = int(0.1 * total_steps)

    cross_encoder.fit(
        train_dataloader=train_dataloader,
        evaluator=None,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    # Evaluation
    val_texts = [(r.prompt, r.response) for r in val_df.itertuples()]
    val_labels = val_df["label"].values
    val_scores = cross_encoder.predict(val_texts)
    val_preds = (val_scores >= 0.5).astype(int)

    metrics = {
        "cross_acc": float(accuracy_score(val_labels, val_preds)),
        "cross_f1": float(f1_score(val_labels, val_preds)),
    }
    try:
        metrics["cross_roc_auc"] = float(roc_auc_score(val_labels, val_scores))
    except Exception:
        metrics["cross_roc_auc"] = None

    cross_encoder.save(str(out_dir))
    Path("metrics").mkdir(exist_ok=True)
    (Path("metrics") / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Cross-Encoder saved to:", out_dir)
    print("📊 Cross-Encoder metrics:", metrics)
    return metrics


# -------------------------
# TRANSFORMER CLASSIFIER
# -------------------------
def train_transformer_classifier(
    train_df, val_df, out_dir: Path,
    model_name: str, max_length: int,
    lr: float, bs_train: int, bs_eval: int,
    epochs: int,
    fast_ci: bool = False,
    max_steps: int = -1,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = build_hf_dataset(train_df)
    val_dataset = build_hf_dataset(val_df)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=True,  # ✅ plus léger que padding="max_length"
        )

    train_tokenized = train_dataset.map(tokenize_fn, batched=True).remove_columns(["text"])
    val_tokenized = val_dataset.map(tokenize_fn, batched=True).remove_columns(["text"])
    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds)),
        }

    # 🔥 Deadline: fast_ci force un entraînement court
    if fast_ci:
        epochs = 1
        if max_steps <= 0:
            max_steps = 30  # ✅ garanti < 5 min

    args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        eval_strategy="no" if fast_ci else "epoch",
        save_strategy="no" if fast_ci else "epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,
        num_train_epochs=epochs,
        max_steps=max_steps,  # ✅ coupe dur
        weight_decay=0.01,
        load_best_model_at_end=False if fast_ci else True,
        metric_for_best_model="f1",
        logging_steps=5 if fast_ci else 50,
        report_to="none",
        fp16=False,  # CPU friendly
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate() if not fast_ci else {}

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # On calcule au moins les metrics sur val même en fast_ci
    if fast_ci:
        preds = trainer.predict(val_tokenized)
        m = compute_metrics((preds.predictions, preds.label_ids))
        eval_metrics = {f"eval_{k}": v for k, v in m.items()}
    else:
        # trainer.evaluate() renvoie souvent eval_accuracy / eval_f1 déjà
        eval_metrics = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in eval_metrics.items()}

    (out_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    print("✅ Transformer classifier saved to:", out_dir)
    print("📊 Transformer metrics:", eval_metrics)
    return eval_metrics, model, tokenizer


# -------------------------
# MLflow: Register from run artifacts (FAST)
# -------------------------
def register_hf_artifact_as_model(
    registered_name: str,
    artifact_subpath: str,
    tags: dict,
    wait_ready: bool = True,
    wait_seconds: int = 60,
):
    """
    Enregistre dans MLflow Model Registry à partir des artefacts du run:
    model_uri = runs:/<run_id>/<artifact_subpath>
    -> Évite mlflow.pyfunc.log_model() (lent sur Windows).
    """
    client = MlflowClient()
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run found for registration.")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/{artifact_subpath}"

    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)

    if wait_ready:
        for _ in range(wait_seconds):
            m = client.get_model_version(name=registered_name, version=mv.version)
            if str(m.status).upper() == "READY":
                break
            time.sleep(1)

    # tags sur la version
    for k, v in (tags or {}).items():
        if v is None:
            continue
        client.set_model_version_tag(registered_name, mv.version, str(k), str(v))

    print(f"✅ Registered model: {registered_name} v{mv.version} from {model_uri}")
    return {"name": registered_name, "version": int(mv.version), "run_id": run_id, "model_uri": model_uri}


# =========================
# MAIN
# =========================
def main():
    set_env_no_wandb()
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    # -------------------------
    # ARGUMENTS
    # -------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--experiment", type=str, default="llm-mlops")
    parser.add_argument("--run_name", type=str, default="baseline")

    # Cross Encoder
    parser.add_argument("--cross_model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross_batch_size", type=int, default=16)
    parser.add_argument("--cross_epochs", type=int, default=1)
    parser.add_argument("--skip_cross", action="store_true")

    # Transformer classifier
    parser.add_argument("--clf_model_name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bs_train", type=int, default=8)
    parser.add_argument("--bs_eval", type=int, default=16)
    parser.add_argument("--clf_epochs", type=int, default=1)

    parser.add_argument("--fast_ci", action="store_true")
    parser.add_argument("--max_steps", type=int, default=30)

    args = parser.parse_args()

    # -------------------------
    # DATA
    # -------------------------
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.experiment)

    registered_name = "llm_transformer_classifier"
    client = MlflowClient()

    # ============================================================
    # RUN 1: CROSS ENCODER (OPTIONAL) + LOG + REGISTER
    # ============================================================
    if not args.skip_cross:
        with mlflow.start_run(run_name=args.run_name + "_cross"):
            mlflow.log_params({
                "cross_model_name": args.cross_model_name,
                "cross_batch_size": args.cross_batch_size,
                "cross_epochs": args.cross_epochs,
                "fast_ci": args.fast_ci,
            })

            cross_metrics = train_cross_encoder(
                train_df=train_df,
                val_df=val_df,
                out_dir=models_dir / "cross_encoder",
                model_name=args.cross_model_name,
                batch_size=args.cross_batch_size,
                epochs=args.cross_epochs,
                fast_ci=args.fast_ci,
            )

            for k, v in cross_metrics.items():
                if v is not None:
                    mlflow.log_metric(k, float(v))

            cross_f1 = cross_metrics.get("eval_f1") or cross_metrics.get("f1")
            cross_acc = cross_metrics.get("eval_accuracy") or cross_metrics.get("accuracy")

            # --- log as MLflow Model (pyfunc) ---
            from sentence_transformers import CrossEncoder

            ce_dir = str(models_dir / "cross_encoder")

            class CrossEncoderPyfunc(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    self.model = CrossEncoder(context.artifacts["ce_dir"])

                def predict(self, context, model_input):
                    pairs = list(zip(
                        model_input.iloc[:, 0].astype(str),
                        model_input.iloc[:, 1].astype(str),
                    ))
                    return np.asarray(self.model.predict(pairs))

            mlflow.pyfunc.log_model(
                artifact_path="ce_model",
                python_model=CrossEncoderPyfunc(),
                artifacts={"ce_dir": ce_dir},
                registered_model_name=registered_name,
            )

            run_id = mlflow.active_run().info.run_id
            versions = client.search_model_versions(f"name='{registered_name}'")
            mv = max([v for v in versions if v.run_id == run_id], key=lambda v: int(v.version))

            if cross_f1 is not None:
                client.set_model_version_tag(registered_name, mv.version, "metric_f1", str(float(cross_f1)))
            if cross_acc is not None:
                client.set_model_version_tag(registered_name, mv.version, "metric_accuracy", str(float(cross_acc)))
            client.set_model_version_tag(registered_name, mv.version, "model_type", "cross_encoder")

            print(f"✅ Registered CROSS_ENCODER v{mv.version} (f1={cross_f1})")

    # ============================================================
    # RUN 2: TRANSFORMER CLASSIFIER + LOG + REGISTER
    # ============================================================
    with mlflow.start_run(run_name=args.run_name + "_classifier"):
        mlflow.log_params({
            "clf_model_name": args.clf_model_name,
            "max_length": args.max_length,
            "lr": args.lr,
            "bs_train": args.bs_train,
            "bs_eval": args.bs_eval,
            "clf_epochs": args.clf_epochs,
            "fast_ci": args.fast_ci,
            "max_steps": args.max_steps,
        })

        clf_metrics, model, tokenizer = train_transformer_classifier(
            train_df=train_df,
            val_df=val_df,
            out_dir=models_dir / "transformer_classifier",
            model_name=args.clf_model_name,
            max_length=args.max_length,
            lr=args.lr,
            bs_train=args.bs_train,
            bs_eval=args.bs_eval,
            epochs=args.clf_epochs,
            fast_ci=args.fast_ci,
            max_steps=args.max_steps,
        )

        for k, v in clf_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        f1 = clf_metrics.get("eval_f1")
        acc = clf_metrics.get("eval_accuracy")
        # --------------------------------------------------
# WRITE TRAIN METRICS FOR DVC
# --------------------------------------------------


        Path("metrics").mkdir(exist_ok=True)

        Path("metrics/train_metrics.json").write_text(
        json.dumps(
            {
                "eval_f1": float(f1) if f1 is not None else None,
                "eval_accuracy": float(acc) if acc is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
        )
        print("✅ Wrote metrics/train_metrics.json for DVC")


        mlt.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            task="text-classification",
            artifact_path="hf_model",
            registered_model_name=registered_name,
        )

        run_id = mlflow.active_run().info.run_id
        versions = client.search_model_versions(f"name='{registered_name}'")
        mv = max([v for v in versions if v.run_id == run_id], key=lambda v: int(v.version))

        if f1 is not None:
            client.set_model_version_tag(registered_name, mv.version, "metric_f1", str(float(f1)))
        if acc is not None:
            client.set_model_version_tag(registered_name, mv.version, "metric_accuracy", str(float(acc)))
        client.set_model_version_tag(registered_name, mv.version, "model_type", "transformer")

        print(f"✅ Registered TRANSFORMER v{mv.version} (f1={f1})")


# =========================
if __name__ == "__main__":
    main()