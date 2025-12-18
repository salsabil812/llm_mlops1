from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REF = Path("monitoring/reference/reference.csv")
CUR = Path("monitoring/current/current.csv")
OUT_HTML = Path("monitoring/reports/evidently_report.html")
OUT_JSON = Path("metrics/evidently.json")

def main():
    if not REF.exists():
        raise FileNotFoundError(f"Missing reference: {REF}")
    if not CUR.exists():
        raise FileNotFoundError(f"Missing current: {CUR}")

    ref = pd.read_csv(REF)
    cur = pd.read_csv(CUR)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    report.save_html(str(OUT_HTML))
    report.save_json(str(OUT_JSON))

    print("✅ Report:", OUT_HTML.resolve())
    print("✅ JSON:", OUT_JSON.resolve())

if __name__ == "__main__":
    main()
