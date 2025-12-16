from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "llm-mlops"

client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=20,
)

print("run_id | runName | status")
print("-" * 80)
for r in runs:
    print(
        f"{r.info.run_id} | {r.data.tags.get('mlflow.runName')} | {r.info.status}"
    )
