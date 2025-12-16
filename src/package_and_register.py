import argparse
from pathlib import Path
import mlflow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="llm-mlops")
    ap.add_argument("--model_dir", default="models/transformer_classifier")
    ap.add_argument("--registered_name", default="llm_transformer_classifier")
    args = ap.parse_args()

    mlflow.set_experiment(args.experiment)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    with mlflow.start_run(run_name="package_model_for_registry"):
        mlflow.set_tag("stage", "package_for_registry")

        # Log an actual MLflow model (creates MLmodel)
        try:
            import transformers  # noqa: F401
            import mlflow.transformers
        except Exception as e:
            raise RuntimeError(
                "Missing mlflow-transformers dependencies. Try:\n"
                "  pip install -U mlflow[transformers] transformers\n"
                f"Error: {e}"
            )

        # Load from local HF directory and log as MLflow model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # This logs a real MLflow model artifact at artifact_path="model"
        mlflow.transformers.log_model(
            transformers_model={"model": mdl, "tokenizer": tok},
            artifact_path="model",
            task="text-classification",
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        print("✅ Logged MLflow model at:", model_uri)

        # Register it
        res = mlflow.register_model(model_uri=model_uri, name=args.registered_name)
        print(f"✅ Registered: {args.registered_name} v{res.version}")

if __name__ == "__main__":
    main()
