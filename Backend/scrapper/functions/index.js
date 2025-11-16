const functions = require("firebase-functions");
const axios = require("axios");

exports.searchJobs = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  const { job, location, type } = req.body;

  const requestData = {
    keywords: job || "",
    location: location || "",
  };

  try {
    const response = await axios.post(
      "https://jooble.org/api/268f120c-4674-4a09-bf3a-5e157592429",
      requestData
    );

    let results = response.data.jobs || [];

    // Filter manually job types (optional)
    if (type) {
      results = results.filter((job) =>
        job.type && job.type.toLowerCase().includes(type)
      );
    }

    res.json(results);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "API failed" });
  }
});
