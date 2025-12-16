# src/train.py
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder, InputExample
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

import mlflow

#log_mlflow
def log_transformer_registry_ready(model_dir: Path, max_length: int = 256):
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    class HFClassifierPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["hf_dir"])
            self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["hf_dir"])
            self.model.eval()
            self.max_length = int(context.model_config.get("max_length", 256))

        def predict(self, context, model_input):
            # Accept DataFrame with 'text' OR prompt/response
            if isinstance(model_input, pd.DataFrame):
                if "text" in model_input.columns:
                    texts = model_input["text"].fillna("").astype(str).tolist()
                elif "prompt" in model_input.columns and "response" in model_input.columns:
                    texts = (
                        "PROMPT: " + model_input["prompt"].fillna("").astype(str)
                        + " RESPONSE: " + model_input["response"].fillna("").astype(str)
                    ).tolist()
                else:
                    texts = model_input.iloc[:, 0].fillna("").astype(str).tolist()
            else:
                texts = [str(x) for x in model_input]

            probs = []
            bs = 8
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    batch = texts[i:i+bs]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                    )
                    logits = self.model(**inputs).logits
                    p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    probs.append(p)

            return np.concatenate(probs)

    mlflow.pyfunc.log_model(
        artifact_path="model",               # IMPORTANT: standard path
        python_model=HFClassifierPyfunc(),
        artifacts={"hf_dir": str(model_dir)},
        model_config={"max_length": int(max_length)},
        pip_requirements=["mlflow", "torch", "transformers", "numpy", "pandas"],
    )

def log_crossencoder_registry_ready(model_dir: Path):
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    from sentence_transformers import CrossEncoder

    class CrossEncoderPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.ce = CrossEncoder(context.artifacts["ce_dir"], num_labels=1)

        def predict(self, context, model_input):
            # Accept DataFrame with prompt/response or 2 columns
            if isinstance(model_input, pd.DataFrame):
                if "prompt" in model_input.columns and "response" in model_input.columns:
                    pairs = list(zip(
                        model_input["prompt"].fillna("").astype(str).tolist(),
                        model_input["response"].fillna("").astype(str).tolist(),
                    ))
                else:
                    cols = list(model_input.columns)
                    pairs = list(zip(
                        model_input[cols[0]].fillna("").astype(str).tolist(),
                        model_input[cols[1]].fillna("").astype(str).tolist(),
                    ))
            else:
                arr = np.asarray(model_input)
                pairs = list(zip(arr[:, 0].astype(str), arr[:, 1].astype(str)))

            scores = self.ce.predict(pairs)
            return np.asarray(scores)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CrossEncoderPyfunc(),
        artifacts={"ce_dir": str(model_dir)},
        pip_requirements=["mlflow", "sentence-transformers", "numpy", "pandas"],
    )

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


# -------------------------
# CROSS ENCODER
# -------------------------
def train_cross_encoder(
    train_df, val_df, out_dir: Path,
    model_name: str, batch_size: int, epochs: int
):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _clean_text_cols(train_df)
    val_df = _clean_text_cols(val_df)

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

    total_steps = math.ceil(len(train_samples) / batch_size) * epochs
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
        "acc": float(accuracy_score(val_labels, val_preds)),
        "f1": float(f1_score(val_labels, val_preds)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(val_labels, val_scores))
    except Exception:
        metrics["roc_auc"] = None

    cross_encoder.save(str(out_dir))
    log_crossencoder_registry_ready(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Cross-Encoder saved to:", out_dir)
    print("📊 Cross-Encoder metrics:", metrics)

    return metrics


# -------------------------
# HF DATASET
# -------------------------
def build_hf_dataset(df: pd.DataFrame):
    df = df.copy()
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    texts = ("PROMPT: " + df["prompt"] + " RESPONSE: " + df["response"]).tolist()
    labels = df["label"].tolist()

    return Dataset.from_dict({"text": texts, "label": labels})



# -------------------------
# TRANSFORMER CLASSIFIER
# -------------------------
def train_transformer_classifier(
    train_df, val_df, out_dir: Path,
    model_name: str, max_length: int,
    lr: float, bs_train: int, bs_eval: int, epochs: int
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
            padding="max_length",
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

    args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
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
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    # Log a registry-ready MLflow model (with MLmodel)
    log_transformer_registry_ready(out_dir, max_length=max_length)

    (out_dir / "metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )

    print("✅ Transformer classifier saved to:", out_dir)
    print("📊 Transformer metrics:", eval_metrics)

    return eval_metrics


# -------------------------
# MAIN
# -------------------------
def main():
    set_env_no_wandb()
    # Disable any hidden autologgers that may log params twice
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass


    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="data/processed/pairwise_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/processed/pairwise_val.csv")
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--experiment", type=str, default="llm-mlops")
    parser.add_argument("--run_name", type=str, default="baseline")

    # Cross Encoder
    parser.add_argument("--cross_model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross_batch_size", type=int, default=16)
    parser.add_argument("--cross_epochs", type=int, default=2)
    parser.add_argument("--skip_cross", action="store_true", help="Skip cross-encoder training")

    # Classifier
    parser.add_argument("--clf_model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bs_train", type=int, default=8)
    parser.add_argument("--bs_eval", type=int, default=16)
    parser.add_argument("--clf_epochs", type=int, default=2)

    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    _validate_df(train_df, "train_df")
    _validate_df(val_df, "val_df")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.experiment)

    # -------------------------
    # RUN 1: CROSS ENCODER
    # -------------------------
    if not args.skip_cross:
        with mlflow.start_run(run_name=args.run_name + "_cross"):
            mlflow.log_params({
                "cross_model_name": args.cross_model_name,
                "cross_batch_size": args.cross_batch_size,
                "cross_epochs": args.cross_epochs,
            })

            cross_metrics = train_cross_encoder(
                train_df=train_df,
                val_df=val_df,
                out_dir=models_dir / "cross_encoder",
                model_name=args.cross_model_name,
                batch_size=args.cross_batch_size,
                epochs=args.cross_epochs,
            )

            for k, v in cross_metrics.items():
                if v is not None:
                    mlflow.log_metric(k, float(v))

            mlflow.log_artifacts(str(models_dir / "cross_encoder"), artifact_path="cross_encoder")

    # -------------------------
    # RUN 2: CLASSIFIER
    # -------------------------
    with mlflow.start_run(run_name=args.run_name + "_classifier"):
        mlflow.log_params({
            "clf_model_name": args.clf_model_name,
            "max_length": args.max_length,
            "lr": args.lr,
            "bs_train": args.bs_train,
            "bs_eval": args.bs_eval,
            "clf_epochs": args.clf_epochs,
        })

        clf_metrics = train_transformer_classifier(
            train_df=train_df,
            val_df=val_df,
            out_dir=models_dir / "transformer_classifier",
            model_name=args.clf_model_name,
            max_length=args.max_length,
            lr=args.lr,
            bs_train=args.bs_train,
            bs_eval=args.bs_eval,
            epochs=args.clf_epochs,
        )

        for k, v in clf_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        mlflow.log_artifacts(
            str(models_dir / "transformer_classifier"),
            artifact_path="transformer_classifier",
        )


if __name__ == "__main__":
    main()
