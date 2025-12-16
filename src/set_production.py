import argparse
from mlflow.tracking import MlflowClient
import mlflow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--alias", default="production")  # "production" alias
    args = ap.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"✅ Set alias '{args.alias}' for {args.model_name} -> v{args.version}")
    print("Tracking URI:", mlflow.get_tracking_uri())

if __name__ == "__main__":
    main()
