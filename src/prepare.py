# src/prepare.py
import os, re, ast, argparse
import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLS = ["prompt", "response_a", "response_b"]

def clean_text_field(x):
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
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\ud800-\udfff]", "", text)

def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    # plus rapide que iterrows sur gros df (mais ok aussi si petit)
    rows = []
    for _, r in df.iterrows():
        rows.append({"prompt": r["prompt"], "response": r["response_a"], "label": 1 if r.get("winner_model_a", 0) == 1 else 0})
        rows.append({"prompt": r["prompt"], "response": r["response_b"], "label": 1 if r.get("winner_model_b", 0) == 1 else 0})
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", default="data/raw/train.csv")
    p.add_argument("--out", dest="out_train", default="data/processed/mini_train.csv")
    p.add_argument("--out_val", dest="out_val", default="data/processed/mini_val.csv")
    p.add_argument("--sample_rows", type=int, default=2000)     # ✅ accélère tout
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_chars", type=int, default=600)
    p.add_argument("--max_response_chars", type=int, default=1200)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.sample_rows > 0 and len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=args.seed).copy()

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text_field).apply(remove_surrogates)

    # remove ties/unknown if columns exist
    if {"winner_model_a","winner_model_b","winner_tie"}.issubset(df.columns):
        df = df[(df["winner_tie"] != 1)].copy()

    pair_df = build_pairwise(df)

    pair_df["prompt"] = pair_df["prompt"].astype(str).apply(remove_surrogates).str.slice(0, args.max_prompt_chars)
    pair_df["response"] = pair_df["response"].astype(str).apply(remove_surrogates).str.slice(0, args.max_response_chars)

    train_df, val_df = train_test_split(pair_df, test_size=args.val_size, random_state=args.seed, stratify=pair_df["label"])

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_val), exist_ok=True)
    train_df.to_csv(args.out_train, index=False, encoding="utf-8")
    val_df.to_csv(args.out_val, index=False, encoding="utf-8")
    print("✅ Saved:", args.out_train, args.out_val)

if __name__ == "__main__":
    main()
