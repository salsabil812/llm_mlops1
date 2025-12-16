import argparse
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/pairwise_train.csv")
    parser.add_argument("--val_csv", default="data/processed/pairwise_val.csv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    def build_text(df):
        return (
            "PROMPT: " + df["prompt"].astype(str)
            + " RESPONSE: " + df["response"].astype(str)
        )

    X_train = build_text(train_df)
    y_train = train_df["label"].astype(int)

    X_val = build_text(val_df)
    y_val = val_df["label"].astype(int)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X_train, y_train)

    train_ds = Dataset(
        pd.DataFrame({"text": X_train, "label": y_train}),
        label="label",
        cat_features=[]
    )

    val_ds = Dataset(
        pd.DataFrame({"text": X_val, "label": y_val}),
        label="label",
        cat_features=[]
    )

    suite = model_evaluation()
    result = suite.run(train_ds, val_ds, pipe)

    result.save_as_html("deepchecks_report.html")

    print("✅ Deepchecks report saved as deepchecks_report.html")


if __name__ == "__main__":
    main()
