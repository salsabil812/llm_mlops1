import mlflow
from pathlib import Path

mlflow.set_experiment("llm-mlops")
with mlflow.start_run(run_name="artifact_upload_test"):
    Path("hello.txt").write_text("hello", encoding="utf-8")
    mlflow.log_artifact("hello.txt")
    rid = mlflow.active_run().info.run_id
    print("run_id:", rid)
