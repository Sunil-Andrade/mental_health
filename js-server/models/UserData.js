const mongoose = require("mongoose");

const userDataSchema = new mongoose.Schema({
    user_id: String,

    q1: Number,
    q2: Number,
    q3: Number,
    q4: Number,
    q5: Number,
    q6: Number,
    q7: Number,
    q8: Number,
    q9: Number,
    q10: Number,

    sleep_hours: Number,
    social_usage: Number,

    self_mood: Number,
    anxiety_level: Number,

    created_at: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model("UserData", userDataSchema);