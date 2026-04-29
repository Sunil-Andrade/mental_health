function computeScore(data) {
    const {
        sleep_hours,
        typing_speed,
        error_rate,
        task_score,
        mood_score,
    } = data;

    let score = 0;

    // Sleep (less sleep = higher risk)
    if (sleep_hours < 6) score += 10;
    else if (sleep_hours < 7) score += 5;

    // Typing speed deviation (simplified)
    if (typing_speed < 50) score += 5;

    // Error rate
    if (error_rate > 0.08) score += 5;

    // Task engagement
    if (task_score < 0.5) score += 5;

    // Mood (1-5 scale, lower = worse)
    if (mood_score <= 2) score += 10;
    else if (mood_score == 3) score += 5;

    return score;
}

module.exports = { computeScore };