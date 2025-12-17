# src/promote.py
import argparse
from mlflow.tracking import MlflowClient


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def pick_best_version(client: MlflowClient, model_name: str, metric_tag: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No model versions found for model '{model_name}'")

    scored = []
    for v in versions:
        tags = getattr(v, "tags", {}) or {}
        score = safe_float(tags.get(metric_tag))
        scored.append((score, v))

    scored_valid = [(s, v) for (s, v) in scored if s is not None]
    if not scored_valid:
        best_v = max(versions, key=lambda x: int(x.version))
        best_score = None
        reason = "no metric tags found -> picked latest version"
    else:
        scored_valid.sort(key=lambda x: x[0], reverse=True)
        best_score, best_v = scored_valid[0]
        reason = f"picked highest {metric_tag}={best_score}"

    return best_v, best_score, reason


def archive_old_production_only(client: MlflowClient, model_name: str, keep_version: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.current_stage == "Production" and str(v.version) != str(keep_version):
            client.transition_model_version_stage(
                name=model_name,
                version=str(v.version),
                stage="Archived",
            )
            print(f"📦 Archived old Production: v{v.version}")


def promote_best(
    model_name: str,
    metric_tag: str = "metric_f1",
    stage: str = "Staging",
    archive_existing_versions: bool = False,
    archive_old_production: bool = False,
):
    client = MlflowClient()

    best_v, best_score, reason = pick_best_version(client, model_name, metric_tag)
    best_version = str(best_v.version)

    model_type = (getattr(best_v, "tags", {}) or {}).get("model_type")
    print(f"📦 model_type={model_type}")
    print(f"🏁 Selection: v{best_version} ({reason})")

    # If requested: archive only old production versions when promoting to Production
    if stage == "Production" and archive_old_production:
        try:
            archive_old_production_only(client, model_name, keep_version=best_version)
        except Exception as e:
            print(f"⚠️ Could not archive old Production versions: {e}")

    promoted = False

    # Try stages
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        promoted = True
        print(f"✅ Promoted {model_name} v{best_version} -> stage={stage} (score={best_score})")
    except Exception as e:
        print(f"⚠️ transition_model_version_stage failed (fallback to alias). Reason: {e}")

    # Fallback aliases (useful in newer MLflow setups)
    if not promoted:
        alias = None
        if stage.lower() == "production":
            alias = "prod"
        elif stage.lower() == "staging":
            alias = "staging"
        elif stage.lower() == "archived":
            alias = "archived"

        if alias:
            try:
                client.set_registered_model_alias(model_name, alias, best_version)
                print(f"✅ Set alias: {model_name}@{alias} -> v{best_version} (score={best_score})")
            except Exception as e:
                print(f"⚠️ set_registered_model_alias failed: {e}")

        # Store stage as tag too (works everywhere)
        try:
            client.set_model_version_tag(model_name, best_version, "stage", stage)
        except Exception:
            pass

    # Always tag promoted + score
    try:
        client.set_model_version_tag(model_name, best_version, "promoted", "true")
    except Exception:
        pass

    if best_score is not None:
        try:
            client.set_model_version_tag(model_name, best_version, metric_tag, str(best_score))
        except Exception:
            pass

    return {"model_name": model_name, "best_version": int(best_version), "best_score": best_score, "stage": stage}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--metric_tag", type=str, default="metric_f1")
    p.add_argument("--stage", type=str, default="Staging", choices=["Staging", "Production", "Archived", "None"])
    p.add_argument("--archive_existing_versions", action="store_true",
                   help="If set, MLflow will archive other versions in same stage automatically.")
    p.add_argument("--archive_old_production", action="store_true",
                   help="If set and stage=Production, archive only the previous Production versions.")
    args = p.parse_args()

    res = promote_best(
        args.model_name,
        metric_tag=args.metric_tag,
        stage=args.stage,
        archive_existing_versions=args.archive_existing_versions,
        archive_old_production=args.archive_old_production,
    )
    print("RESULT:", res)


if __name__ == "__main__":
    main()
