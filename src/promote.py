# src/promote.py
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "llm-mlops"

REGISTRY = {
    "transformer_classifier": {
        "registered_name": "llm_transformer_classifier",
        "artifact_path": "transformer_classifier",  # preferred name (auto-find anyway)
    },
    "cross_encoder": {
        "registered_name": "llm_cross_encoder",
        "artifact_path": "cross_encoder",
    },
}


def _get_experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise RuntimeError(f"Experiment '{name}' not found")
    return exp.experiment_id


def _deepchecks_passed_source_run_ids(client: MlflowClient, exp_id: str) -> set[str]:
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="tags.stage = 'deepchecks' and metrics.deepchecks_pass >= 1",
        max_results=200,
    )
    ok = set()
    for r in runs:
        sid = r.data.tags.get("source_run_id")
        if sid:
            ok.add(sid)
    return ok


def _pick_best_eval_run(
    client: MlflowClient,
    exp_id: str,
    model_type: str,
    allowed_source_ids: set[str] | None,
):
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.stage = 'evaluation' and tags.model_type = '{model_type}'",
        max_results=200,
    )

    best = None
    best_key = None

    for r in runs:
        sid = r.data.tags.get("source_run_id")
        if not sid:
            continue
        if allowed_source_ids is not None and sid not in allowed_source_ids:
            continue

        f1 = r.data.metrics.get("eval_f1")
        if f1 is None:
            continue

        auc = r.data.metrics.get("eval_auc", -1.0)
        lat = r.data.metrics.get("eval_latency_ms_per_sample", 1e9)

        # maximize f1, then auc, then minimize latency
        key = (float(f1), float(auc), -float(lat))

        if best is None or key > best_key:
            best = r
            best_key = key

    return best, best_key


# -----------------------------
# Artifact path discovery
# -----------------------------
def _list_artifacts(client: MlflowClient, run_id: str, path: str = ""):
    return client.list_artifacts(run_id, path)


def _find_artifact_dir(client: MlflowClient, run_id: str, candidates: list[str]) -> str:
    """
    Return the first existing artifact directory among candidates.
    Searches top-level and one-level deep.
    """
    top = _list_artifacts(client, run_id, "")
    top_paths = {a.path: a for a in top}

    # direct match at top-level
    for c in candidates:
        a = top_paths.get(c)
        if a is not None and a.is_dir:
            return c

    # one-level deep: parent/candidate
    for a in top:
        if not a.is_dir:
            continue
        children = _list_artifacts(client, run_id, a.path)
        child_paths = {ch.path: ch for ch in children}
        for c in candidates:
            nested = f"{a.path}/{c}"
            ch = child_paths.get(nested)
            if ch is not None and ch.is_dir:
                return nested

    raise RuntimeError(
        f"Could not find any artifact dir in run {run_id} among candidates={candidates}. "
        f"Top-level artifacts: {[x.path for x in top]}"
    )


def _download_model_artifacts_auto(
    client: MlflowClient,
    source_run_id: str,
    dst_dir: Path,
    candidates: list[str],
) -> Path:
    artifact_path = _find_artifact_dir(client, source_run_id, candidates)
    local_path = client.download_artifacts(
        run_id=source_run_id,
        path=artifact_path,
        dst_path=str(dst_dir),
    )
    return Path(local_path)


# -----------------------------
# Log as REAL registrable MLflow Models (pyfunc)
# -----------------------------
def _log_transformer_as_pyfunc(local_hf_dir: Path, max_length: int = 256):
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class HFClassifierPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["hf_dir"])
            self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["hf_dir"])
            self.model.eval()
            self.max_length = int(context.model_config.get("max_length", 256))

        def predict(self, context, model_input):
            # Expect DataFrame with column "text" OR prompt/response
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
                    batch = texts[i:i + bs]
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
        artifact_path="model",
        python_model=HFClassifierPyfunc(),
        artifacts={"hf_dir": str(local_hf_dir)},
        model_config={"max_length": int(max_length)},
        pip_requirements=[
            "mlflow",
            "torch",
            "transformers",
            "numpy",
            "pandas",
        ],
    )


def _log_crossencoder_as_pyfunc(local_ce_dir: Path):
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    from sentence_transformers import CrossEncoder

    class CrossEncoderPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.ce = CrossEncoder(context.artifacts["ce_dir"], num_labels=1)

        def predict(self, context, model_input):
            if isinstance(model_input, pd.DataFrame):
                if "prompt" in model_input.columns and "response" in model_input.columns:
                    pairs = list(zip(
                        model_input["prompt"].astype(str).tolist(),
                        model_input["response"].astype(str).tolist(),
                    ))
                else:
                    cols = list(model_input.columns)
                    pairs = list(zip(
                        model_input[cols[0]].astype(str).tolist(),
                        model_input[cols[1]].astype(str).tolist(),
                    ))
            else:
                arr = np.asarray(model_input)
                pairs = list(zip(arr[:, 0].astype(str), arr[:, 1].astype(str)))

            scores = self.ce.predict(pairs)
            return np.asarray(scores)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CrossEncoderPyfunc(),
        artifacts={"ce_dir": str(local_ce_dir)},
    )


def _wait_until_model_visible(client: MlflowClient, run_id: str, retries: int = 8, sleep_s: int = 5):
    """
    DagsHub/remote artifact stores can be eventually consistent.
    We wait until 'model/' is visible and contains MLmodel / python_model.pkl, etc.
    """
    last_top = []
    last_model = []
    for i in range(retries):
        top = client.list_artifacts(run_id, "")
        model = client.list_artifacts(run_id, "model")
        last_top = [a.path for a in top]
        last_model = [a.path for a in model]

        if len(model) > 0:
            return last_top, last_model

        time.sleep(sleep_s)

    return last_top, last_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=EXPERIMENT_NAME)
    ap.add_argument("--require_deepchecks", action="store_true", default=False)
    ap.add_argument(
        "--force_model",
        choices=["transformer", "cross"],
        default=None,
        help="Optional: force winner selection for debugging",
    )
    args = ap.parse_args()

    client = MlflowClient()
    exp_id = _get_experiment_id(client, args.experiment)

    allowed_source_ids = None
    if args.require_deepchecks:
        ok = _deepchecks_passed_source_run_ids(client, exp_id)
        if not ok:
            raise RuntimeError("No Deepchecks PASS run found (deepchecks_pass >= 1).")
        allowed_source_ids = ok

    best_tr, key_tr = _pick_best_eval_run(client, exp_id, "transformer_classifier", allowed_source_ids)
    best_ce, key_ce = _pick_best_eval_run(client, exp_id, "cross_encoder", allowed_source_ids)

    if best_tr is None and best_ce is None:
        raise RuntimeError("No eligible evaluation runs found. Run evaluate_min.py first.")

    candidates = []
    if best_tr is not None:
        candidates.append(("transformer_classifier", best_tr, key_tr))
    if best_ce is not None:
        candidates.append(("cross_encoder", best_ce, key_ce))

    if args.force_model == "transformer":
        winner_type, winner_eval_run, _ = [c for c in candidates if c[0] == "transformer_classifier"][0]
    elif args.force_model == "cross":
        winner_type, winner_eval_run, _ = [c for c in candidates if c[0] == "cross_encoder"][0]
    else:
        winner_type, winner_eval_run, _ = max(candidates, key=lambda x: x[2])

    source_run_id = winner_eval_run.data.tags["source_run_id"]
    registered_name = REGISTRY[winner_type]["registered_name"]

    print(f"🏆 Winner: {winner_type}")
    print(f"   eval_f1={winner_eval_run.data.metrics.get('eval_f1')}")
    print(f"   eval_auc={winner_eval_run.data.metrics.get('eval_auc')}")
    print(f"   source_run_id={source_run_id}")
    print(f"📦 Packaging winner into an MLflow Model, then registering as: {registered_name}")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="package_and_register"):
        mlflow.set_tag("stage", "package_for_registry")
        mlflow.set_tag("winner_model_type", winner_type)
        mlflow.set_tag("source_run_id", source_run_id)
        mlflow.set_tag("eval_run_id", winner_eval_run.info.run_id)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            if winner_type == "transformer_classifier":
                find_candidates = ["transformer_classifier", "clf", "models/transformer_classifier", "transformer"]
            else:
                find_candidates = ["cross_encoder", "cross", "models/cross_encoder"]

            local_dir = _download_model_artifacts_auto(client, source_run_id, td_path, find_candidates)
            print("✅ Downloaded artifacts to:", local_dir)

            if winner_type == "transformer_classifier":
                _log_transformer_as_pyfunc(local_dir, max_length=256)
            else:
                _log_crossencoder_as_pyfunc(local_dir)

            # Force creation/visibility of model/ dir on remote artifact store
            Path("model_marker.txt").write_text("ok", encoding="utf-8")
            mlflow.log_artifact("model_marker.txt", artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        print("✅ Logged MLflow model at:", model_uri)

        # Wait until remote store shows model/ artifacts
        c2 = MlflowClient()
        top_paths, model_paths = _wait_until_model_visible(c2, run_id, retries=10, sleep_s=5)
        print("Artifacts in packaging run:", top_paths)
        print("Artifacts in model/:", model_paths)

        if not model_paths:
            raise RuntimeError(
                "Model artifacts are not visible under 'model/' in this run (remote store delay or upload issue). "
                "Open the run in DagsHub -> Artifacts and confirm that 'model/MLmodel' exists. "
                "If it exists, register manually from the UI."
            )

        res = mlflow.register_model(model_uri=model_uri, name=registered_name)
        print(f"✅ Registered: {registered_name} v{res.version}")


if __name__ == "__main__":
    main()
