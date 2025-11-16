# quality score (grammar+cv pages)
# relevence
RELEVENCE_API_KEY = (
    "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"
)

API_FEEDBACK = (
    "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"
)

MODEL_NAME = "alibaba/tongyi-deepresearch-30b-a3b:free"


job_title = "Machine Learning Engineer"

Job_title = "full-stack developer"


import json
import re
from openai import OpenAI
import language_tool_python
import logging
from json_repair import repair_json

import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

"""### import json file"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(ch):
    ch = ch.replace("json", "")
    ch = ch.replace("```", "")
    ch = re.sub(r"\s+", " ", ch)
    ch = ch.replace("None", "null")
    ch = ch.encode("ascii", "ignore").decode("ascii")
    return ch.strip()


failed_feedback = {
    "score": None,
    "feedback": {"weaknesses": "", "strengths": "", "suggestions": ""},
}

"""### compute relevance score"""


def compute_relevance(cv_text: str, job_title: str, api_key: str):

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_message = """
You are an expert resume evaluator.
YOUR ONLY OUTPUT MUST BE VALID JSON — NOTHING ELSE.
No explanations, no markdown, no text before or after JSON.

BE VERY STRICT:
- Only give high relevance scores if the CV truly matches the job title.
- Penalize missing important skills, missing sections (experience, skills, projects), or irrelevant content.
- Rarely assign a score of 100.

REQUIRED FORMAT:
{
  "score": float
}

RULES:
1. JSON must be valid and parseable by Python.
2. Ensure the relevance_score is never inflated and reflects missing skills or sections.
"""

    user_message = f"Job Title: {job_title}\n\nCV:\n{cv_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )

        raw_text = response.choices[0].message.content.strip()
        print(raw_text)
        logging.debug(f"Raw model response: {raw_text}")

        try:
            raw_text = repair_json(raw_text)
            raw_text = clean_text(raw_text)
            result = json.loads(raw_text)
            if "score" not in result:
                raise ValueError("Missing 'score' field in model response.")
            logging.info(f"Relevance score computed successfully: {result['score']}")
            return result
        except json.JSONDecodeError:
            logging.error("Invalid JSON returned by model.")
            return {"score": None}
        except ValueError as ve:
            logging.error(str(ve))
            return {"score": None}

    except Exception as e:
        logging.exception("OpenAI API error occurred.")
        return {"score": None}
    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return {"score": None}


"""### Quality score"""


def concat_dict(value):
    parts = []

    if isinstance(value, dict):
        for v in value.values():
            concatenated = concat_dict(v)
            if concatenated:
                parts.append(concatenated)
    elif isinstance(value, list):
        for v in value:
            concatenated = concat_dict(v)
            if concatenated:
                parts.append(concatenated)
    else:
        if value is not None:
            parts.append(str(value))

    return " ".join(parts)


def compute_quality_score(data):
    tool = language_tool_python.LanguageTool("en-US")

    text = concat_dict(data)
    print("Flattened text:", text)
    matches = tool.check(text)
    print("Number of grammar issues:", len(matches))

    try:
        text_length = len(str(data))
        error_ratio = len(matches) / max(text_length, 1)
        pages = data.get("pages", 1)  # default to 1 page if missing
        page_penalty = max(0.0, 1 - 0.05 * (pages - 1))
        quality_score = (1 - error_ratio) * page_penalty
        logging.info(f"Quality score computed: {quality_score:.4f}")
    except Exception as e:
        logging.exception("Failed to compute quality score.")
        quality_score = 0.0
    return quality_score


"""### Section Score

CV_Score=sections​+quality​+relevance​
CV Score∈[0,100]
🛑 Personal Information
🛑 Website and Social Links
💡 Professional Summaries
🛑 Work Experience
🛑 Education
💡 Certification
💡 Awards and Achievement
💡 Projects
🛑 Skills and Interests
💡 Volunteering
💡 Publication
"""


class SectionScore:

    SECTIONS_WEIGHT = {
        "personal_information": 2,
        "website_and_social_links": 2,
        "professional_summary": 1,
        "work_experience": 3,
        "education": 2,
        "certification": 1,
        "awards_and_achievements": 1,
        "projects": 1,
        "skills_and_interests": 2,
        "volunteering": 1,
        "publications": 0.5,
    }

    TOTAL_WEIGHT: float = 17.5

    def __init__(self) -> None:
        self.sections_score = 0.0

    def is_empty(self, value):
        """Recursively check if a value is empty."""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, list):
            return all(self.is_empty(v) for v in value)
        if isinstance(value, dict):
            return all(self.is_empty(v) for v in value.values())
        return False

    def check_sections(self, section_name: str, section_data):
        if self.is_empty(section_data):
            return False

        if section_name in ["work_experience", "education"]:
            return isinstance(section_data, list) and len(section_data) >= 1

        if section_name == "skills_and_interests":
            # Require at least 3 items and optionally check languages
            if isinstance(section_data, dict):
                if "languages" in section_data and section_data["languages"]:
                    return len(section_data.get("skills", [])) >= 3
                return False
            if isinstance(section_data, list):
                return len(section_data) >= 3
            return False

        if section_name in ["certification", "projects"]:
            return isinstance(section_data, list) and len(section_data) >= 2

        if section_name == "professional_summary":
            return isinstance(section_data, str) and len(section_data.strip()) >= 50

        # Default: section exists and is not empty
        return True

    def compute_sections_score(self, cv):

        self.sections_score = 0.0
        validation_results = {}

        for section_name, weight in self.SECTIONS_WEIGHT.items():
            section_data = cv.get(section_name)
            is_valid = self.check_sections(section_name, section_data)
            validation_results[section_name] = {
                "valid": is_valid,
                "weight": weight,
                "data_present": not self.is_empty(section_data),
            }
            if is_valid:
                self.sections_score += weight
                logging.info(f"✅ {section_name}: +{weight} points")
            else:
                logging.info(f"❌ {section_name}: 0 points (weight: {weight})")

        score_percentage = (self.sections_score / self.TOTAL_WEIGHT) * 100

        return {
            "score_percentage": round(score_percentage, 2),
            "raw_score": round(self.sections_score, 2),
            "total_weight": self.TOTAL_WEIGHT,
            "validation_details": validation_results,
            "missing_sections": [
                name
                for name, details in validation_results.items()
                if not details["valid"]
            ],
        }


"""### Total score"""


def compute_overall_cv_score(
    quality_score, relevance_score, section_score, weights=None
):

    if weights is None:
        weights = {"quality": 0.2, "relevance": 0.4, "sections": 0.4}

    quality_norm = min(max(quality_score, 0.0), 1.0)
    relevance_norm = min(max(relevance_score.get("score", 0.0) / 100.0, 0.0), 1.0)
    sections_norm = min(
        max(section_score.get("score_percentage", 0.0) / 100.0, 0.0), 1.0
    )

    # Compute weighted score
    overall_score = (
        weights["quality"] * quality_norm
        + weights["relevance"] * relevance_norm
        + weights["sections"] * sections_norm
    )

    logging.info(f"Overall CV score computed: {overall_score:.4f}")
    return overall_score


def get_cv_score(data, job_title, api):
    section_score = SectionScore()
    section_score = section_score.compute_sections_score(data)
    quality_score = compute_quality_score(data)
    relevance_score = compute_relevance(data, job_title, api)
    weights = {"quality": 0.2, "relevance": 0.4, "sections": 0.4}
    cv_score = (
        compute_overall_cv_score(quality_score, relevance_score, section_score, weights)
        * 100
    )
    print(f"Overall CV score: {cv_score:.4f}")

    return cv_score


"""### feedback"""


def generate_feedback(cv_text: str, job_title: str, api_key: str):

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_message = """
You are an expert resume reviewer with deep knowledge of recruitment, hiring trends, and candidate evaluation.

TASK:
- Analyze the provided CV thoroughly in the context of the given job title.
- Focus on three aspects: weaknesses, strengths, and actionable suggestions.
- Provide concrete, specific, and professional feedback.

OUTPUT REQUIREMENTS:
- Your output MUST be valid JSON and NOTHING ELSE.
- Do NOT add any explanations, commentary, or markdown.
- JSON format MUST be exactly:

{
    "weaknesses": "string describing candidate weaknesses relevant to the job",
    "strengths": "string describing candidate strengths relevant to the job",
    "suggestions": "string with clear, actionable advice to improve the CV"
}

RULES:
- Avoid generic statements; be specific to the CV content.
- Mention gaps, missing skills, or formatting issues in weaknesses.
- Highlight relevant achievements, skills, and experience in strengths.
- Give practical, actionable steps to improve the CV in suggestions.
- Ensure the JSON is always parseable by Python's json.loads().
"""

    user_message = f"Job Title: {job_title}\n\nCV:\n{cv_text}"

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-9b-v2:free",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        raw_text = response.choices[0].message.content.strip()
        logging.debug(
            f"Raw feedback response: {raw_text[:500]}..."
        )  # preview first 500 chars

        try:
            feedback_dict = json.loads(raw_text)
            for key in ["weaknesses", "strengths", "suggestions"]:
                if key not in feedback_dict:
                    feedback_dict[key] = ""
            logging.info("Feedback successfully generated.")
            return feedback_dict
        except json.JSONDecodeError:
            logging.error("Invalid JSON returned by feedback model.")
            return {
                "weaknesses": "",
                "strengths": "",
                "suggestions": "",
                "raw_response": raw_text,
            }

    except Exception as e:
        logging.exception("OpenAI API error occurred.")
        return {"weaknesses": "", "strengths": "", "suggestions": ""}
    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return {"weaknesses": "", "strengths": "", "suggestions": ""}


from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import json, logging

app = FastAPI(
    title="CV Review API",
    description="Analyzes a CV and returns strengths, weaknesses, and suggestions.",
    version="1.0.0",
)

# --- Example dependencies ---
# You should have these functions implemented elsewhere in your project:
# from your_module import get_cv_score, generate_feedback


class CVInput(BaseModel):
    cv_json: dict
    job_title: str


@app.post("/review", summary="Analyze a CV and provide review feedback")
def review_cv(
    payload: CVInput = Body(
        ...,
        example={
            "cv_json": {"name": "John Doe", "skills": ["Python", "Data Analysis"]},
            "job_title": "Data Engineer",
        },
    )
):
    """
    Analyze a candidate's CV for a specific job title.

    - **cv_json**: Parsed CV content in JSON format
    - **job_title**: The target job title to evaluate fit for

    Returns:
    A JSON object containing:
    - **weaknesses**: Areas to improve
    - **strengths**: Key strengths in the CV
    - **suggestions**: Practical improvement suggestions
    - **cv_score**: Rounded overall score
    """

    try:
        cv_score = get_cv_score(payload.cv_json, payload.job_title, RELEVENCE_API_KEY)
        feedback = generate_feedback(payload.cv_json, payload.job_title, API_FEEDBACK)

        review = {
            "weaknesses": feedback.get("weaknesses", ""),
            "strengths": feedback.get("strengths", ""),
            "suggestions": feedback.get("suggestions", ""),
            "cv_score": round(cv_score),
        }

        logging.info("CV review completed successfully.")

        return review

    except Exception as e:
        logging.error(f"Error during CV review: {e}")
        raise HTTPException(status_code=500, detail=str(e))
