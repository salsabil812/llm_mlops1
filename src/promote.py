# src/promote.py
import argparse
from mlflow.tracking import MlflowClient


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def get_versions(client, model_name: str):
    return client.search_model_versions(f"name='{model_name}'")


def get_current_production(client, model_name: str):
    for v in get_versions(client, model_name):
        if v.current_stage == "Production":
            return v
    return None


def get_version_by_number(client, model_name: str, version: str):
    for v in get_versions(client, model_name):
        if str(v.version) == str(version):
            return v
    return None


def get_latest_version(client, model_name: str):
    versions = get_versions(client, model_name)
    if not versions:
        return None
    return max(versions, key=lambda x: int(x.version))


def get_metric(v, metric_tag: str):
    tags = getattr(v, "tags", {}) or {}
    return safe_float(tags.get(metric_tag))


def archive_old_production_only(client, model_name: str, keep_version: str):
    for v in get_versions(client, model_name):
        if v.current_stage == "Production" and str(v.version) != str(keep_version):
            client.transition_model_version_stage(model_name, str(v.version), "Archived")
            print(f"📦 Archived old Production: v{v.version}")


def transition(client, model_name: str, version: str, stage: str, archive_existing_versions: bool = False):
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def promote_auto(
    model_name: str,
    metric_tag: str,
    candidate: str = "latest",  # "latest" or explicit version number
    archive_old_production: bool = True,
):
    client = MlflowClient()

    cand_v = get_latest_version(client, model_name) if candidate == "latest" else get_version_by_number(client, model_name, candidate)
    if cand_v is None:
        raise RuntimeError(f"Candidate not found for model '{model_name}' (candidate={candidate}).")

    prod_v = get_current_production(client, model_name)

    cand_score = get_metric(cand_v, metric_tag)
    prod_score = get_metric(prod_v, metric_tag) if prod_v else None

    print(f"🧪 Candidate: v{cand_v.version} score({metric_tag})={cand_score}")
    print(f"🏭 Production: {('none' if not prod_v else f'v{prod_v.version} score({metric_tag})={prod_score}')}")

    # If no production exists yet => candidate becomes production
    if prod_v is None:
        transition(client, model_name, cand_v.version, "Production")
        print(f"✅ No existing Production. Promoted v{cand_v.version} -> Production")
        return

    # If metrics missing, be conservative: send candidate to Staging
    if cand_score is None or prod_score is None:
        transition(client, model_name, cand_v.version, "Staging")
        print("⚠️ Missing metric(s). Promoted candidate -> Staging (Production unchanged).")
        return

    # Decision rule you requested
    if cand_score > prod_score:
        if archive_old_production:
            archive_old_production_only(client, model_name, keep_version=str(cand_v.version))
        transition(client, model_name, cand_v.version, "Production")
        print(f"✅ Candidate better than Production ({cand_score} > {prod_score}). v{cand_v.version} -> Production")
    else:
        transition(client, model_name, cand_v.version, "Staging")
        print(f"✅ Candidate NOT better ({cand_score} <= {prod_score}). v{cand_v.version} -> Staging (Production unchanged).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--metric_tag", default="metric_f1")

    # Modes:
    p.add_argument("--mode", choices=["auto", "manual"], default="manual")

    # manual mode (your old behavior)
    p.add_argument("--stage", choices=["Staging", "Production", "Archived", "None"], default="Staging")
    p.add_argument("--archive_existing_versions", action="store_true")

    # auto mode
    p.add_argument("--candidate", default="latest", help="latest or a version number")
    p.add_argument("--archive_old_production", action="store_true")

    args = p.parse_args()

    client = MlflowClient()

    if args.mode == "auto":
        promote_auto(
            model_name=args.model_name,
            metric_tag=args.metric_tag,
            candidate=args.candidate,
            archive_old_production=args.archive_old_production,
        )
        return

    # manual mode: keep your old behavior (promote the best global to the requested stage)
    # (If you still want that, you can keep your original pick_best_version logic here.)
    latest = get_latest_version(client, args.model_name)
    if latest is None:
        raise RuntimeError(f"No model versions found for '{args.model_name}'")

    transition(client, args.model_name, latest.version, args.stage, archive_existing_versions=args.archive_existing_versions)
    print(f"✅ Manual: promoted v{latest.version} -> {args.stage}")


if __name__ == "__main__":
    main()
