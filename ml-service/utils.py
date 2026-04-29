import numpy as np



def questionnaire_score(sample):
    q = sample[:10]  # q1–q10
    return sum(q) / 10 

# ---------- FEATURE EXTRACTION ----------
def extract_features(record):
    return [
        record["q1"], record["q2"], record["q3"], record["q4"], record["q5"],
        record["q6"], record["q7"], record["q8"], record["q9"], record["q10"],
        record["sleep_hours"],
        record["social_usage"],
        record["self_mood"],
        record["anxiety_level"]
    ]


# ---------- GOOD DAY DETECTION ----------
def is_good_day(record):
    return (
        record["self_mood"] >= 4 and
        record["anxiety_level"] <= 4 and
        record["sleep_hours"] >= 6
    )


# ---------- BASELINE FROM GOOD DAYS ----------
def compute_baseline(history):
    
    good_days = []

    for h in history:
        if (
            h["self_mood"] >= 4 and
            h["anxiety_level"] <= 4 and
            h["sleep_hours"] >= 6
        ):
            good_days.append(extract_features(h))

    # ✅ GOOD CASE
    if len(good_days) >= 3:
        return np.mean(good_days, axis=0)

    # ❌ NO GOOD DAYS → FORCE SAFE BASELINE
    # instead of mean(history)

    baseline = np.mean([extract_features(h) for h in history], axis=0)

    # override key features with safe defaults
    baseline[10] = max(baseline[10], 6)   # sleep should not go below 6
    baseline[12] = max(baseline[12], 3)   # mood neutral
    baseline[13] = min(baseline[13], 4)   # anxiety cap

    return baseline


# ---------- Z-SCORE DEVIATION ----------
def compute_deviation(sample, baseline):
    sample = np.array(sample)
    baseline = np.array(baseline)

    diff = np.abs(sample - baseline)

# scale down
    return float(np.mean(diff))


# ---------- TREND ----------
def compute_trend(history_vectors):
    

    if len(history_vectors) < 3:
        return 0

    X = np.array(history_vectors)

    # focus on key features only (sleep, mood, anxiety)
    sleep = X[:, 10]
    mood = X[:, 12]
    anxiety = X[:, 13]

    # compute simple slope (last - first)
    sleep_trend = sleep[-1] - sleep[0]
    mood_trend = mood[-1] - mood[0]
    anxiety_trend = anxiety[-1] - anxiety[0]

    # combine (note: bad trends should increase score)
    trend_score = (
    (-sleep_trend if sleep_trend < 0 else 0) * 2 +
    (-mood_trend if mood_trend < 0 else 0) * 2 +
    (anxiety_trend if anxiety_trend > 0 else 0) * 2
    )

    return trend_score


# ---------- SHOCK ----------
def shock_score(sample, baseline):
    sleep_index = 10

    drop = baseline[sleep_index] - sample[sleep_index]

    if drop >= 3:
        return 10
    elif drop >= 2:
        return 5
    return 0


# ---------- DURATION ----------
def duration_penalty(history, baseline):
    sleep_index = 10
    count = 0

    for h in reversed(history):
        features = extract_features(h)
        if abs(features[sleep_index] - baseline[sleep_index]) > 2:
            count += 1
        else:
            break

    return count * 2