import express from "express";
import cors from "cors";
import fetch from "node-fetch";
// Load environment variables from a local .env file in development.
// This file should NOT be committed. Use the example file `backend/.env.example`.
import 'dotenv/config';

const app = express();
app.use(cors());
app.use(express.json());

// Read the Jooble API key from environment. Do NOT store real keys in source.
const API_KEY = process.env.JOOBLE_API_KEY || "";
const JOOBLE_API_URL = `https://jooble.org/api/${API_KEY}`;

if (!API_KEY) {
    console.warn("⚠️  Warning: JOOBLE_API_KEY is not set. Requests to Jooble will likely fail.");
}

/* --------------------------
   HTML ENTITY DECODER
--------------------------- */
function decodeHTML(str = "") {
    return str
        .replace(/&nbsp;/g, " ")
        .replace(/&#[0-9]+;/g, "")
        .replace(/&amp;/g, "&")
        .replace(/&quot;/g, '"')
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">");
}

/* --------------------------
   CLEAN JOB DESCRIPTION
--------------------------- */
function extractSections(raw) {
    if (!raw) return {};

    let text = decodeHTML(raw);

    // Remove HTML tags
    text = text.replace(/<\/?[^>]+(>|$)/g, "");

    // Normalize bullet points + spacing
    text = text
        .replace(/\r\n/g, "\n")
        .replace(/~+/g, "• ")
        .replace(/\n{2,}/g, "\n\n")
        .trim();

    const sections = {
        companyInfo: "",
        jobMission: "",
        skills: ""
    };

    const skillKeywords = [
        "requirements",
        "skills",
        "qualifications",
        "what you bring",
        "needed"
    ];

    const missionKeywords = [
        "responsibilities",
        "what you will do",
        "mission",
        "tasks",
        "role description"
    ];

    let lower = text.toLowerCase();

    // Extract Skills
    for (const k of skillKeywords) {
        if (lower.includes(k)) {
            const idx = lower.indexOf(k);
            sections.skills = text.slice(idx).trim();
            text = text.slice(0, idx).trim();
            lower = text.toLowerCase();
            break;
        }
    }

    // Extract Mission
    for (const k of missionKeywords) {
        if (lower.includes(k)) {
            const idx = lower.indexOf(k);
            sections.jobMission = text.slice(idx).trim();
            text = text.slice(0, idx).trim();
            lower = text.toLowerCase();
            break;
        }
    }

    // Whatever remains = company intro
    sections.companyInfo = text.trim();

    return sections;
}

/* --------------------------
         API ROUTE
--------------------------- */
app.post("/search", async (req, res) => {
    try {
        const { job, location } = req.body;

        const response = await fetch(JOOBLE_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                keywords: job || "",
                location: location || ""
            }),
        });

        const text = await response.text();

        let data;
        try {
            data = JSON.parse(text);
        } catch {
            return res.status(500).json({
                error: "Invalid JSON returned from Jooble",
                details: text
            });
        }

        // Clean each job description
        const jobs = (data.jobs || []).map(job => ({
            ...job,
            cleaned: extractSections(job.snippet || job.description || "")
        }));

        res.json({
            totalCount: data.totalCount || 0,
            jobs
        });

    } catch (error) {
        res.status(500).json({
            error: "Server error",
            details: error.message,
        });
    }
});

/* --------------------------
          START SERVER
--------------------------- */
app.listen(3001, () => {
    console.log("🔥 Server running on http://localhost:3001");
});
