const express = require("express");
const router = express.Router();
const UserData = require("../models/UserData");
const { getPrediction } = require("../services/model");

router.post("/", async (req, res) => {
    try {
        const data = req.body;

        console.log("requested")

        // 1. Save current data
        await UserData.create(data);

        // 2. Fetch history (last 20)
        const history = await UserData.find({ user_id: data.user_id })
            .sort({ created_at: -1 })
            .limit(20)
            .lean();

        // 3. Reverse for chronological order
        history.reverse();

        // 4. Call ML service
        const result = await getPrediction(data.user_id, history, data);

        res.json(result);

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Server error" });
    }
});

module.exports = router;