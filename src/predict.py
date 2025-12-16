# src/predict.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow


def build_text(df: pd.DataFrame) -> list[str]:
    return (
        "PROMPT: " + df["prompt"].fillna("").astype(str)
        + " RESPONSE: " + df["response"].fillna("").astype(str)
    ).tolist()


def normalize_scores(raw_out, n: int) -> np.ndarray:
    """
    Make a 1D score array from various possible pyfunc outputs.
    Your promoted model (pyfunc) should return probabilities for class=1 (float).
    """
    if isinstance(raw_out, np.ndarray):
        arr = raw_out
    else:
        try:
            arr = np.asarray(raw_out)
        except Exception:
            arr = raw_out

    # if pandas
    if hasattr(raw_out, "to_numpy"):
        # DataFrame/Series
        try:
            if hasattr(raw_out, "columns") and "score" in raw_out.columns:
                return raw_out["score"].to_numpy(dtype=float)
            return raw_out.to_numpy(dtype=float).reshape(-1)
        except Exception:
            pass

    # list-like
    try:
        arr = np.array(raw_out, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError("Wrong output length")
        return arr
    except Exception:
        raise RuntimeError(
            f"Model output format not supported. Got type={type(raw_out)}. "
            "Expected an array-like of float scores."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV must contain columns prompt,response")
    ap.add_argument("--output_csv", default="predictions.csv")

    ap.add_argument("--model_uri", default=None, help="e.g. models:/llm_transformer_classifier/1")
    ap.add_argument("--model_name", default="llm_transformer_classifier")
    ap.add_argument("--model_version", default="1")

    ap.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    df = pd.read_csv(inp)
    if not {"prompt", "response"}.issubset(df.columns):
        raise ValueError("input_csv must contain columns: prompt, response")

    texts = build_text(df)
    model_uri = args.model_uri or f"models:/{args.model_name}/{args.model_version}"

    print("✅ Loading model from:", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    # We send a "text" column (most convenient for inference)
    model_input = pd.DataFrame({"text": texts})

    raw_out = model.predict(model_input)
    scores = normalize_scores(raw_out, n=len(df))

    df["score"] = scores
    df["pred"] = (df["score"] >= args.threshold).astype(int)

    outp = Path(args.output_csv)
    df.to_csv(outp, index=False)
    print("✅ Saved:", outp.resolve())
    print("✅ Rows:", len(df))


if __name__ == "__main__":
    main()
