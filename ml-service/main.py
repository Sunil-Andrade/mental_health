from fastapi import FastAPI
from schemas import Data
from model import UserModel, final_score, get_status
from utils import (
    extract_features,
    compute_baseline,
    compute_deviation,
    compute_trend,
    shock_score,
    duration_penalty
)

app = FastAPI()


@app.post("/predict")
def predict(data: Data):

    # ---------- CLEAN HISTORY ----------
    required_keys = [
        "q1","q2","q3","q4","q5",
        "q6","q7","q8","q9","q10",
        "sleep_hours","social_usage",
        "self_mood","anxiety_level"
    ]

    history_records = [
        h for h in data.history
        if all(k in h for k in required_keys)
    ]

    history_vectors = [extract_features(h) for h in history_records]
    sample = extract_features(data.current)

    print("CURRENT SAMPLE:", sample)

    print("RAW HISTORY:", len(data.history))
    print("VALID HISTORY:", len(history_records))

    # ---------- LOW DATA ----------
    if len(history_vectors) < 5:
        return {
            "status": "Collecting Data",
            "score": 0
        }

    # ---------- MODEL ----------
    model = UserModel()
    model.train(history_vectors)
    anomaly = model.predict(sample)

    # ---------- BASELINE ----------
    baseline = compute_baseline(history_records)

    # ---------- METRICS ----------
    deviation = compute_deviation(sample, history_vectors)
    trend = compute_trend(history_vectors)
    shock = shock_score(sample, baseline)
    duration = duration_penalty(history_records, baseline)

    # ---------- FINAL ----------
    score = final_score(anomaly, deviation, trend, shock, duration)
    status = get_status(score)

    return {
        "status": status,
        "score": round(score, 2),
        "debug": {
            "anomaly": float(anomaly),
            "deviation": float(deviation),
            "trend": float(trend),
            "shock": float(shock),
            "duration": float(duration)
        }
    }