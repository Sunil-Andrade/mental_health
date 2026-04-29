from fastapi import FastAPI
from schemas import Data
from model import UserModel, final_score, get_status
from utils import compute_baseline, compute_deviation, compute_trend,shock_score

app = FastAPI()


def extract_features(record):
    return [
        record["q1"],
        record["q2"],
        record["q3"],
        record["q4"],
        record["q5"],
        record["q6"],
        record["q7"],
        record["q8"],
        record["q9"],
        record["q10"],
        record["sleep_hours"],
        record["social_usage"],
        record["self_mood"],
        record["anxiety_level"]
    ]


@app.post("/predict")
def predict(data: Data):

    # --------- PREPARE HISTORY ---------
    history_vectors = [extract_features(h) for h in data.history]

    sample = extract_features(data.current)

    # --------- HANDLE LOW DATA ---------
    if len(history_vectors) < 5:
        return {
            "status": "Collecting Data",
            "score": 0
        }

    # --------- MODEL ---------
    model = UserModel()
    model.train(history_vectors)

    anomaly = model.predict(sample)

    # --------- BASELINE ---------
    baseline = compute_baseline(history_vectors)

    # --------- DEVIATION ---------
    deviation = compute_deviation(sample, baseline)

    # --------- TREND ---------
    trend = compute_trend(history_vectors)


    #----------- Shoock --------
    

    # --------- FINAL SCORE ---------
    score = final_score(anomaly, deviation, trend)

    

    status = get_status(score)

    return {
        "status": status,
        "score": round(score, 2),
        "debug": {
            "anomaly": anomaly,
            "deviation": deviation,
            "trend": trend
        }
    }