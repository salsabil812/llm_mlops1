import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/pairwise.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Ensure required columns exist
    needed = {"prompt", "response", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "pairwise_train.csv"
    val_path = out / "pairwise_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("✅ Saved:", train_path)
    print("✅ Saved:", val_path)

if __name__ == "__main__":
    main()
