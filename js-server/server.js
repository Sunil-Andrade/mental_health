const mongoose = require("mongoose");
const app = require("./app");

const PORT = 5000;

mongoose.connect("mongodb://127.0.0.1:27017/mental_app")
  .then(() => {
    console.log("MongoDB connected");

    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  })
  .catch(err => console.error(err));