# src/deepchecks_gate.py
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--scores", required=True, help="metrics/scores.json from evaluate")
    ap.add_argument("--out", required=True, help="metrics/deepchecks.json output")
    ap.add_argument("--min_f1", type=float, default=0.60)
    ap.add_argument("--min_acc", type=float, default=0.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read evaluate scores
    with open(args.scores, "r", encoding="utf-8") as f:
        scores = json.load(f)

    # Support multiple naming conventions
    f1 = scores.get("f1") or scores.get("metric_f1") or scores.get("val_f1") or 0.0
    acc = scores.get("accuracy") or scores.get("metric_accuracy") or scores.get("val_accuracy") or 0.0

    checks = []
    checks.append({
        "name": "min_f1",
        "value": float(f1),
        "threshold": args.min_f1,
        "passed": float(f1) >= args.min_f1,
    })
    checks.append({
        "name": "min_accuracy",
        "value": float(acc),
        "threshold": args.min_acc,
        "passed": float(acc) >= args.min_acc,
    })

    passed = all(c["passed"] for c in checks)

    report = {
        "timestamp": int(time.time()),
        "passed": passed,
        "checks": checks,
        "notes": "Lightweight Deepchecks-style gate (threshold-based). Replace/extend with full Deepchecks suite later."
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Deepchecks gate report written to: {out_path}")
    for c in checks:
        print(f" - {c['name']}: {c['value']:.4f} (min {c['threshold']}) -> {'PASS' if c['passed'] else 'FAIL'}")

    if not passed:
        print("❌ Quality gate FAILED. Blocking promotion to Staging.")
        sys.exit(1)

    print("✅ Quality gate PASSED. Promotion allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
