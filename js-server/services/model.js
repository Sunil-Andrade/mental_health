const axios = require("axios");

const ML_URL = "http://13.232.136.66:8000/predict";

async function getPrediction(user_id, history, current) {
    const response = await axios.post(ML_URL, {
        user_id,
        history,
        current
    });

    return response.data;
}

module.exports = { getPrediction };