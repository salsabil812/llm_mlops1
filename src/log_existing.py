import json
from pathlib import Path
import mlflow


def log_metrics_json(p: Path, prefix: str):
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    for k, v in d.items():
        if isinstance(v, (int, float)) and v is not None:
            mlflow.log_metric(f"{prefix}_{k}", float(v))


def main():
    mlflow.set_experiment("llm-mlops")

    cross = Path("models/cross_encoder")
    clf = Path("models/transformer_classifier")

    print("Local exists?")
    print(" - cross:", cross.exists(), "->", cross)
    print(" - clf  :", clf.exists(), "->", clf)

    with mlflow.start_run(run_name="import_existing_models_v2"):
        if cross.exists():
            mlflow.log_artifacts(str(cross), artifact_path="cross_encoder")
            log_metrics_json(cross / "metrics.json", "cross")

        if clf.exists():
            mlflow.log_artifacts(str(clf), artifact_path="transformer_classifier")
            log_metrics_json(clf / "metrics.json", "clf")

        # show what's actually logged at top-level
        from mlflow.tracking import MlflowClient
        c = MlflowClient()
        rid = mlflow.active_run().info.run_id
        top = c.list_artifacts(rid, "")
        print("✅ Imported. Run:", rid)
        print("✅ Top-level artifacts:", [a.path for a in top])


if __name__ == "__main__":
    main()
