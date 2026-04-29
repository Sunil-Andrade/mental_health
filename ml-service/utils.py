import numpy as np


def shock_score(sample, baseline):
        drop = baseline[0] - sample[0]  # sleep drop

        if drop >= 3:   # big drop
         return 10   # strong penalty
        elif drop >= 2:
         return 5
        else:
            return 0

def compute_baseline(history):
    X = np.array(history)
    return np.mean(X, axis=0)

def compute_deviation(sample, baseline):
    weights = [
        1.2,1.2,1.2,1.2,1.2,
        1.2,1.2,1.2,1.2,1.2,   # questionnaire

        2.0,  # sleep
        1.5,  # social usage

        2.5,  # self mood
        2.5   # anxiety
    ] 
    
    dev = [abs(s - b) for s, b in zip(sample, baseline)]
    
    return sum(d * w for d, w in zip(dev, weights))

def compute_trend(history):
    if len(history) < 10:
        return 0

    recent = np.array(history[-5:])
    past = np.array(history[-10:-5])

    trend_vector = recent.mean(axis=0) - past.mean(axis=0)

    return np.sum(trend_vector)