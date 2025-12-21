# src/app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mlflow

# =================================================
# Config
# =================================================
MODEL_NAME = "llm_transformer_classifier"
MODEL_VERSION = "1"  # ou "Production"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

app = Flask(__name__)

# =================================================
# Load model ONCE (avoid double-load with Flask reloader)
# =================================================
print("🚀 Loading MLflow model:", MODEL_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Model loaded")


# =================================================
# Helpers
# =================================================
def build_text(prompt: str, response: str) -> str:
    return f"PROMPT: {prompt} RESPONSE: {response}"


def normalize_scores(raw_out, n: int) -> np.ndarray:
    """
    Normalize MLflow pyfunc outputs to a 1D float array of length n.
    Handles outputs like:
      - pandas DataFrame/Series
      - numpy array / list
      - shape (n,2) probabilities -> takes proba of class=1
    """
    if hasattr(raw_out, "to_numpy"):
        arr = raw_out.to_numpy()
    else:
        arr = np.asarray(raw_out)

    arr = np.array(arr)

    # If outputs (n, 2+), take probability of class=1
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, 1]

    arr = arr.reshape(-1)

    if arr.shape[0] != n:
        raise RuntimeError(f"Model output length mismatch: got {arr.shape[0]}, expected {n}")

    return arr.astype(float)


def json_error(message: str, required=None, status_code: int = 400):
    payload = {"error": message}
    if required is not None:
        payload["required"] = required
    return jsonify(payload), status_code


# =================================================
# Home (avoid 404 in browser)
# =================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "llm_mlops1 Flask API",
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "uri": MODEL_URI},
        "endpoints": {
            "GET /": "API info",
            "GET /health": "health check",
            "POST /predict_single": {"prompt": "str", "response": "str"},
            "POST /predict_pair": {"prompt": "str", "response_a": "str", "response_b": "str"},
        },
        "examples": {
            "predict_single": {
                "prompt": "Hello",
                "response": "Hi there"
            },
            "predict_pair": {
                "prompt": "Hello",
                "response_a": "Answer A",
                "response_b": "Answer B"
            }
        }
    })


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    # Return 204 No Content (silences browser favicon requests)
    return ("", 204)


# =================================================
# Endpoint 1 — Single scoring
# =================================================
@app.route("/predict_single", methods=["POST"])
def predict_single():
    """
    Input:  { "prompt": "...", "response": "..." }
    Output: { "score": float, "pred": 0|1 }
    """
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return json_error("Invalid JSON body")

    if not {"prompt", "response"}.issubset(data):
        return json_error("Missing fields", required=["prompt", "response"])

    prompt = str(data.get("prompt", ""))
    response = str(data.get("response", ""))

    df = pd.DataFrame({"text": [build_text(prompt, response)]})
    raw_out = model.predict(df)
    score = float(normalize_scores(raw_out, n=1)[0])

    return jsonify({
        "score": round(score, 6),
        "pred": int(score >= 0.5),
        "label": "chosen" if score >= 0.5 else "rejected"
    })


# =================================================
# Endpoint 2 — Pairwise Kaggle-style
# =================================================
@app.route("/predict_pair", methods=["POST"])
def predict_pair():
    """
    Input:
      { "prompt": "...", "response_a": "...", "response_b": "..." }
    Output:
      { "score_a": float, "score_b": float, "winner_pred": "A" | "B" | "TIE" }
    """
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return json_error("Invalid JSON body")

    required = {"prompt", "response_a", "response_b"}
    if not required.issubset(data):
        return json_error("Missing fields", required=sorted(list(required)))

    prompt = str(data.get("prompt", ""))
    response_a = str(data.get("response_a", ""))
    response_b = str(data.get("response_b", ""))

    df = pd.DataFrame({
        "text": [
            build_text(prompt, response_a),
            build_text(prompt, response_b),
        ]
    })

    raw_out = model.predict(df)
    scores = normalize_scores(raw_out, n=2)
    score_a, score_b = float(scores[0]), float(scores[1])

    # tie tolerance to avoid "tie never happens" due to floats
    eps = 1e-9
    if abs(score_a - score_b) <= eps:
        winner = "TIE"
    elif score_a > score_b:
        winner = "A"
    else:
        winner = "B"

    return jsonify({
        "score_a": round(score_a, 6),
        "score_b": round(score_b, 6),
        "winner_pred": winner
    })


# =================================================
# Health check
# =================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =================================================
# Run server
# =================================================
if __name__ == "__main__":
    # ✅ Important: avoid double artifact download
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
