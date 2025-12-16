# src/prepare.py
import os
import re
import ast
import argparse
import pandas as pd


TEXT_COLS = ["prompt", "response_a", "response_b"]


def clean_text_field(x):
    # handles: list, string that looks like ["a","b"], normal strings, NaN
    if isinstance(x, list):
        return " ".join([str(t) for t in x])
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return " ".join([str(t) for t in parsed])
            except Exception:
                return s
        return s
    if pd.isna(x):
        return ""
    return str(x)


def remove_surrogates(text: str) -> str:
    # remove invalid unicode surrogate chars that can crash pyarrow/utf-8
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\ud800-\udfff]", "", text)


def compute_winner(row):
    # returns: 1 if A won, 0 if B won, -1 if tie/unknown
    if row.get("winner_model_a", 0) == 1:
        return 1
    if row.get("winner_model_b", 0) == 1:
        return 0
    if row.get("winner_tie", 0) == 1:
        return -1
    return -1


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "prompt": r["prompt"],
                "response": r["response_a"],
                "label": 1 if r.get("winner_model_a", 0) == 1 else 0,
            }
        )
        rows.append(
            {
                "prompt": r["prompt"],
                "response": r["response_b"],
                "label": 1 if r.get("winner_model_b", 0) == 1 else 0,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output", default="data/processed/pairwise.csv")
    parser.add_argument("--max_prompt_chars", type=int, default=600)
    parser.add_argument("--max_response_chars", type=int, default=1200)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    print("Raw shape:", df.shape)

    # Clean text columns
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .apply(clean_text_field)
                .apply(remove_surrogates)
            )

    # Build target + filter ties
    df["winner"] = df.apply(compute_winner, axis=1)
    df = df[df["winner"] != -1].copy()
    print("After removing ties:", df.shape)
    print("Winner distribution:\n", df["winner"].value_counts(normalize=True))

    # Build pairwise dataset
    pair_df = build_pairwise(df)

    # Safety sanitize
    pair_df["prompt"] = pair_df["prompt"].astype(str).apply(remove_surrogates)
    pair_df["response"] = pair_df["response"].astype(str).apply(remove_surrogates)

    # Truncate BEFORE saving
    pair_df["prompt"] = pair_df["prompt"].str.slice(0, args.max_prompt_chars)
    pair_df["response"] = pair_df["response"].str.slice(0, args.max_response_chars)

    print("Pairwise shape:", pair_df.shape)
    print("Label distribution:\n", pair_df["label"].value_counts(normalize=True))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pair_df.to_csv(args.output, index=False, encoding="utf-8")
    print("✅ Saved:", args.output)


if __name__ == "__main__":
    main()
