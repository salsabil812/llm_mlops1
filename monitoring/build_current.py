import json
from pathlib import Path
import pandas as pd

LOG_PATH = Path("logs/predictions.jsonl")
OUT_PATH = Path("monitoring/current/current.csv")

def main(n_last: int = 500):
    rows = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    rows = rows[-n_last:]
    df = pd.DataFrame(rows)

    for c in ["ts","prompt","response","score","pred"]:
        if c not in df.columns:
            df[c] = ""

    keep = [c for c in ["ts","prompt","response","score","pred","model_name","model_stage"] if c in df.columns]
    df = df[keep]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("✅ current saved:", OUT_PATH.resolve(), "rows=", len(df))

if __name__ == "__main__":
    main()
