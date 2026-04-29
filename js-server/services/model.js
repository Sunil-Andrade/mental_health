const axios = require("axios");

const ML_URL = "http://127.0.0.1:8000/predict";

async function getPrediction(user_id, history, current) {
    const response = await axios.post(ML_URL, {
        user_id,
        history,
        current
    });

    return response.data;
}

module.exports = { getPrediction };