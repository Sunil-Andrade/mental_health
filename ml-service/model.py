import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class UserModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=0.2,
            random_state=42
        )

    def train(self, data):
        X = np.array(data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, sample):
        sample = np.array(sample).reshape(1, -1)
        sample_scaled = self.scaler.transform(sample)

        return self.model.decision_function(sample_scaled)[0]


# ---------- FINAL SCORE ----------
def final_score(anomaly, deviation, trend, shock, duration,q_score):
    return (
        (-anomaly * 4) +
        (deviation * 1.0) +
        (trend * 2) +
        shock +
        duration +
        (q_score * 5)   # 🔥 strong signal
    )


# ---------- STATUS ----------
def get_status(score):
    if score < 8:
        return "Stable"
    elif score < 18:
        return "Slight Change"
    else:
        return "Needs Attention"